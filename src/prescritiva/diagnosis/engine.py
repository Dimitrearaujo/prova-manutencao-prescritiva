"""Motor de diagnostico: orquestra similaridade, cobertura e geracao.

O fluxo tem tres portoes, nesta ordem, e cada um pode encerrar o diagnostico:

1. O evento se parece com algo do historico? Se nao, e padrao desconhecido e a
   resposta e essa, sem palpite de defeito.
2. O padrao encontrado e um defeito ou um estado de operacao? O enunciado lista
   normal, baseline, teste, acelerando e motor_desligado como nao-problemas, e
   prescrever correcao para eles seria erro.
3. Existe procedimento cadastrado para esse defeito? Se nao, o sistema informa
   que o problema ainda nao esta documentado e pede o cadastro, em vez de
   improvisar instrucao.

So depois dos tres portoes o modelo de linguagem e chamado, e mesmo ali ele so
recebe o texto recuperado, nunca a tarefa de decidir o defeito.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from prescritiva.config import load_catalog, load_settings
from prescritiva.data.ingest import rpm_regime
from prescritiva.data.schema import UNIT_DUPLICATES, VibrationEvent
from prescritiva.diagnosis.historico import EstatisticaHistorica, estatisticas
from prescritiva.knowledge.store import BaseConhecimento, TrechoRecuperado
from prescritiva.llm.base import GeradorTexto
from prescritiva.llm.factory import construir_gerador
from prescritiva.similarity.index import IndiceSimilaridade, ResultadoSimilaridade

INSTRUCAO = """Voce e um assistente de manutencao industrial. Sua unica fonte e o
PROCEDIMENTO fornecido no contexto.

Regras que nao podem ser quebradas:
- Use exclusivamente o texto do procedimento. Nao acrescente etapa, ferramenta,
  tolerancia ou valor numerico que nao esteja escrito ali.
- O tipo de defeito ja foi determinado pela analise de similaridade. Nao
  rediscuta o diagnostico e nao proponha outro defeito.
- Se o procedimento nao cobrir algum ponto, escreva que o procedimento nao trata
  desse ponto. Nunca preencha a lacuna por conta propria.
- Responda em portugues do Brasil, direto ao tecnico que vai executar.

Formato da resposta:
1. O que verificar primeiro (2 a 4 itens)
2. Como corrigir (passos na ordem do procedimento)
3. Como validar depois da correcao"""


@dataclass
class Diagnostico:
    situacao: str
    fault: str | None
    rotulo: str | None
    confianca: float
    regime_rpm: str
    e_defeito: bool
    mensagem: str
    similaridade: dict[str, Any] = field(default_factory=dict)
    cobertura: dict[str, Any] = field(default_factory=dict)
    historico: dict[str, Any] = field(default_factory=dict)
    trechos: list[dict[str, Any]] = field(default_factory=list)
    instrucoes: str = ""
    gerador: str = ""


class MotorDiagnostico:
    def __init__(
        self,
        indice: IndiceSimilaridade,
        base: BaseConhecimento,
        database: Path,
        gerador: GeradorTexto,
        *,
        top_k: int = 4,
    ) -> None:
        self.indice = indice
        self.base = base
        self.database = database
        self.gerador = gerador
        self.top_k = top_k
        catalogo = load_catalog()
        self.catalogo = catalogo["faults"]
        self.estados = catalogo["estados_operacionais"]

    @classmethod
    def carregar(cls) -> "MotorDiagnostico":
        settings = load_settings()
        return cls(
            indice=IndiceSimilaridade.carregar(settings.paths.index_dir),
            base=BaseConhecimento.carregar(settings.paths.index_dir),
            database=settings.paths.database,
            gerador=construir_gerador(settings.llm),
            top_k=settings.knowledge["retrieval_top_k"],
        )

    def diagnosticar(
        self, evento: VibrationEvent | dict[str, Any], *, ignorar_proprio_id: bool = True
    ) -> Diagnostico:
        """Diagnostica um evento.

        `ignorar_proprio_id` importa ao demonstrar com um registro tirado do
        proprio historico: sem ele o vizinho mais proximo e o proprio evento, a
        distancia zero, e o resultado parece bom por um motivo que nao existe em
        producao. Um evento novo de verdade nao tem id conhecido e o parametro
        nao faz diferenca.
        """
        validado = evento if isinstance(evento, VibrationEvent) else VibrationEvent(**evento)
        quadro = _para_dataframe(validado)
        excluir = {validado.id} if ignorar_proprio_id and validado.id is not None else None
        similaridade = self.indice.consultar(quadro, excluir_ids=excluir)

        if not similaridade.reconhecido:
            return self._padrao_desconhecido(similaridade)

        fault = similaridade.fault_predominante
        if fault in self.estados:
            return self._estado_operacional(fault, similaridade)
        return self._defeito(fault, similaridade)

    def _padrao_desconhecido(self, similaridade: ResultadoSimilaridade) -> Diagnostico:
        return Diagnostico(
            situacao="padrao_desconhecido",
            fault=None,
            rotulo=None,
            confianca=similaridade.confianca,
            regime_rpm=similaridade.regime_rpm,
            e_defeito=False,
            mensagem=(
                "Este evento nao corresponde a nenhum padrao do historico operacional. "
                f"{similaridade.motivo} Registre a condicao observada em campo para que "
                "ela passe a fazer parte da base."
            ),
            similaridade=_resumo_similaridade(similaridade),
        )

    def _estado_operacional(self, fault: str, similaridade: ResultadoSimilaridade) -> Diagnostico:
        rotulo = self.estados[fault]["rotulo"]
        return Diagnostico(
            situacao="estado_operacional",
            fault=fault,
            rotulo=rotulo,
            confianca=similaridade.confianca,
            regime_rpm=similaridade.regime_rpm,
            e_defeito=False,
            mensagem=(
                f"O evento corresponde a {rotulo.lower()}, que e uma condicao de operacao "
                "e nao um problema. Nenhuma acao corretiva se aplica."
            ),
            similaridade=_resumo_similaridade(similaridade),
            historico=asdict(estatisticas(self.database, fault)),
        )

    def _defeito(self, fault: str, similaridade: ResultadoSimilaridade) -> Diagnostico:
        entrada = self.catalogo.get(fault, {"rotulo": fault, "termo_chave": fault, "termos_busca": fault})
        rotulo = entrada["rotulo"]
        estatistica = estatisticas(self.database, fault)
        cobertura = self.base.cobertura(entrada["termo_chave"], entrada["termos_busca"], rotulo)

        base_diagnostico = {
            "fault": fault,
            "rotulo": rotulo,
            "confianca": similaridade.confianca,
            "regime_rpm": similaridade.regime_rpm,
            "e_defeito": True,
            "similaridade": _resumo_similaridade(similaridade),
            "cobertura": asdict(cobertura),
            "historico": asdict(estatistica),
        }

        if not cobertura.coberto:
            return Diagnostico(
                situacao="defeito_sem_documentacao",
                mensagem=(
                    f"O padrao identificado e {rotulo}, com {_frase_historico(estatistica)}. "
                    f"Porem nao existe procedimento cadastrado para esse defeito. {cobertura.motivo} "
                    "Cadastre um documento orientativo para que o sistema passe a instruir a correcao."
                ),
                **base_diagnostico,
            )

        trechos = self.base.buscar(
            f"{rotulo} {entrada['termos_busca']} correcao procedimento",
            top_k=self.top_k,
            documento=cobertura.documento,
        )
        instrucoes = self.gerador.gerar(INSTRUCAO, _montar_pergunta(rotulo, estatistica, trechos))

        return Diagnostico(
            situacao="defeito_documentado",
            mensagem=(
                f"O padrao identificado e {rotulo}, com {_frase_historico(estatistica)}. "
                f"{cobertura.motivo}"
            ),
            trechos=[asdict(t) for t in trechos],
            instrucoes=instrucoes,
            gerador=self.gerador.nome,
            **base_diagnostico,
        )

    def perguntar(self, pergunta: str, *, documento: str | None = None) -> dict[str, Any]:
        """Consulta livre a base documental, usada pelo chat."""
        trechos = self.base.buscar(pergunta, top_k=self.top_k, documento=documento)
        if not trechos:
            return {
                "resposta": (
                    "Nenhum procedimento cadastrado trata desse assunto. "
                    "Envie um documento orientativo sobre o tema para que ele passe a ser consultavel."
                ),
                "trechos": [],
                "gerador": "-",
            }
        contexto = "\n---\n".join(f"({t.documento} / {t.secao})\n{t.texto}" for t in trechos)
        resposta = self.gerador.gerar(
            INSTRUCAO,
            f"PERGUNTA: {pergunta}\n\nTRECHOS DO PROCEDIMENTO:\n{contexto}",
        )
        return {
            "resposta": resposta,
            "trechos": [asdict(t) for t in trechos],
            "gerador": self.gerador.nome,
        }


def _para_dataframe(evento: VibrationEvent) -> pd.DataFrame:
    dados = evento.model_dump()
    dados.pop("fault", None)
    dados.pop("id", None)
    for coluna in UNIT_DUPLICATES:
        dados.pop(coluna, None)
    quadro = pd.DataFrame([dados])
    quadro["regime_rpm"] = quadro["rpm"].map(rpm_regime)
    return quadro


def _resumo_similaridade(similaridade: ResultadoSimilaridade) -> dict[str, Any]:
    return {
        "distribuicao": similaridade.distribuicao,
        "distancia_media": similaridade.distancia_media,
        "limiar_rejeicao": similaridade.limiar_rejeicao,
        "reconhecido": similaridade.reconhecido,
        "motivo": similaridade.motivo,
        "vizinhos": [asdict(v) for v in similaridade.vizinhos[:10]],
    }


def _frase_historico(estatistica: EstatisticaHistorica) -> str:
    if not estatistica.ocorrencias:
        return "nenhum registro anterior no historico"
    total = f"{estatistica.ocorrencias:,}".replace(",", ".")
    return (
        f"{total} ocorrencias registradas entre {(estatistica.primeira or '')[:10]} e "
        f"{(estatistica.ultima or '')[:10]}, media de {estatistica.ocorrencias_por_dia:.0f} por dia"
    )


def _montar_pergunta(
    rotulo: str, estatistica: EstatisticaHistorica, trechos: list[TrechoRecuperado]
) -> str:
    contexto = "\n---\n".join(f"({t.documento} / {t.secao})\n{t.texto}" for t in trechos)
    return (
        f"DEFEITO IDENTIFICADO: {rotulo}\n"
        f"HISTORICO: {_frase_historico(estatistica)}.\n\n"
        f"TRECHOS DO PROCEDIMENTO:\n{contexto}"
    )
