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

A consulta livre do chat passa pelo portao 3 tambem, por caminho proprio: sem
isso a solucao recusaria prescrever para um defeito sem procedimento na aba de
diagnostico e prescreveria para o mesmo defeito na aba de consulta, com o texto
do documento de outro defeito.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from prescritiva.config import load_catalog, load_settings
from prescritiva.data.ingest import (
    REGIME_FORA_DA_GRADE,
    RPM_CONHECIDOS,
    regime_aproximado,
    rpm_regime,
)
from prescritiva.data.schema import UNIT_DUPLICATES, VibrationEvent
from prescritiva.diagnosis.contexto import contexto_regime
from prescritiva.diagnosis.historico import EstatisticaHistorica, estatisticas
from prescritiva.knowledge.store import BaseConhecimento, Cobertura, TrechoRecuperado
from prescritiva.llm.base import GeradorTexto
from prescritiva.llm.deterministico import GeradorDeterministico
from prescritiva.llm.factory import construir_gerador
from prescritiva.similarity.index import IndiceSimilaridade, ResultadoSimilaridade
from prescritiva.text import stems

LOGGER = logging.getLogger(__name__)

# Janela temporal excluida da busca por vizinhos, em segundos. O ensaio gravou
# uma leitura a cada dois segundos (mediana medida no parquet), entao cinco
# minutos sao ~150 leituras: duas ordens de grandeza acima do espacamento, e o
# suficiente para que nenhum "gemeo" do proprio evento entre como vizinho.
JANELA_VIZINHANCA_S = 300.0

# Piso de aderencia BM25 para a consulta livre. O filtro anterior era "score > 0",
# que nao filtra nada: qualquer pergunta com uma palavra em comum com o corpus
# devolvia trecho e o modelo escrevia procedimento em cima dele.
PISO_SCORE_CHAT = 2.5

SEM_ASSUNTO = (
    "Nenhum procedimento cadastrado trata desse assunto. Envie um documento orientativo "
    "sobre o tema para que ele passe a ser consultavel."
)

# Titulos de secao que indicam acao, e nao definicao. Os seis procedimentos
# seguem a mesma estrutura: primeiro conceituam o defeito, depois ensinam a
# corrigir. Para prescrever, a segunda metade e que interessa.
SECOES_DE_ACAO = (
    "procedimento correcao execucao etapas passos ferramentas instrumentos "
    "verificacao inspecao ajuste alinhamento montagem validacao manutencao "
    "acoes recomendacoes boas praticas criterios aceitacao"
)

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

# O chat precisa de instrucao propria. A INSTRUCAO do diagnostico manda o modelo
# nao rediscutir um defeito que "ja foi determinado pela analise de
# similaridade" - na consulta livre nao existe diagnostico nenhum, e a frase
# convida o modelo a assumir um defeito que ninguem determinou.
INSTRUCAO_CHAT = """Voce e um assistente de manutencao industrial respondendo a
consulta de um tecnico. Sua unica fonte sao os TRECHOS DO PROCEDIMENTO no
contexto.

Regras que nao podem ser quebradas:
- Use exclusivamente o texto dos trechos. Nao acrescente etapa, ferramenta,
  tolerancia ou valor numerico que nao esteja escrito ali.
- Nao diagnostique a maquina e nao levante hipotese de defeito. Quem determina o
  defeito e a analise de similaridade, nao esta conversa.
- Se os trechos nao responderem a pergunta, escreva que o procedimento cadastrado
  nao trata desse ponto e pare. Nunca preencha a lacuna por conta propria.
- Responda em portugues do Brasil, direto ao tecnico que vai executar, citando o
  documento e a secao de onde veio cada informacao."""


@dataclass
class Diagnostico:
    situacao: str
    fault: str | None
    rotulo: str | None
    confianca: float
    regime_rpm: str
    e_defeito: bool
    mensagem: str
    # A rotacao bruta e o aviso de encaixe viajam com o diagnostico porque
    # regime_rpm sozinho esconde que 1900 rpm foi comparado com o historico de
    # 2000 rpm. Quem consome pela API, pela fila ou pelo CMMS nao tem outra
    # forma de ver isso - ate aqui o aviso existia so na tela do Streamlit.
    rpm: float | None = None
    regime_aproximado: bool = False
    similaridade: dict[str, Any] = field(default_factory=dict)
    cobertura: dict[str, Any] = field(default_factory=dict)
    historico: dict[str, Any] = field(default_factory=dict)
    contexto_operacional: dict[str, Any] = field(default_factory=dict)
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
        janela_vizinhanca_s: float = JANELA_VIZINHANCA_S,
    ) -> None:
        self.indice = indice
        self.base = base
        self.database = database
        self.gerador = gerador
        self.top_k = top_k
        self.janela_vizinhanca_s = janela_vizinhanca_s
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
        self,
        evento: VibrationEvent | dict[str, Any],
        *,
        janela_vizinhanca_s: float | None = None,
    ) -> Diagnostico:
        """Diagnostica um evento.

        `janela_vizinhanca_s` diz quanto tempo em torno do evento sai do indice
        antes da busca, e importa ao demonstrar com um registro tirado do
        proprio historico. Excluir so o id nao resolve: a leitura gravada dois
        segundos depois continua no indice, a distancia praticamente zero, e
        carrega o consenso sozinha. Um evento novo de verdade nao tem gemeo no
        passado, entao a janela nao muda nada para ele - ela existe para que a
        demonstracao rode contra o mesmo sistema que foi medido.
        """
        validado = evento if isinstance(evento, VibrationEvent) else VibrationEvent(**evento)
        if rpm_regime(validado.rpm) == REGIME_FORA_DA_GRADE:
            return _rotacao_sem_historico(validado.rpm)

        quadro = _para_dataframe(validado)
        similaridade = self.indice.consultar(
            quadro,
            excluir_ids={validado.id} if validado.id is not None else None,
            momento=validado.created_at,
            janela_s=self.janela_vizinhanca_s if janela_vizinhanca_s is None else janela_vizinhanca_s,
        )
        aproximado = regime_aproximado(validado.rpm)

        if not similaridade.reconhecido:
            return self._padrao_desconhecido(similaridade, validado.rpm, aproximado)

        fault = similaridade.fault_predominante
        if fault in self.estados:
            return self._estado_operacional(fault, similaridade, validado.rpm, aproximado)
        return self._defeito(fault, similaridade, validado.rpm, aproximado)

    def _padrao_desconhecido(
        self, similaridade: ResultadoSimilaridade, rpm: float, aproximado: bool
    ) -> Diagnostico:
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
                + _aviso_encaixe(rpm, similaridade.regime_rpm, aproximado)
            ),
            rpm=rpm,
            regime_aproximado=aproximado,
            similaridade=_resumo_similaridade(similaridade),
        )

    def _estado_operacional(
        self, fault: str, similaridade: ResultadoSimilaridade, rpm: float, aproximado: bool
    ) -> Diagnostico:
        rotulo = self.estados[fault]["rotulo"]
        estatistica = estatisticas(self.database, fault)
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
                + _aviso_encaixe(rpm, similaridade.regime_rpm, aproximado)
            ),
            rpm=rpm,
            regime_aproximado=aproximado,
            similaridade=_resumo_similaridade(similaridade),
            historico=asdict(estatistica),
            contexto_operacional=asdict(
                contexto_regime(estatistica.por_regime, similaridade.regime_rpm, rotulo)
            ),
        )

    def _defeito(
        self, fault: str, similaridade: ResultadoSimilaridade, rpm: float, aproximado: bool
    ) -> Diagnostico:
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
            "rpm": rpm,
            "regime_aproximado": aproximado,
            "similaridade": _resumo_similaridade(similaridade),
            "cobertura": asdict(cobertura),
            "historico": asdict(estatistica),
            "contexto_operacional": asdict(
                contexto_regime(estatistica.por_regime, similaridade.regime_rpm, rotulo)
            ),
        }

        if not cobertura.coberto:
            return Diagnostico(
                situacao="defeito_sem_documentacao",
                mensagem=(
                    f"O padrao identificado e {rotulo}, com {_frase_historico(estatistica)}. "
                    f"Porem nao existe procedimento cadastrado para esse defeito. {cobertura.motivo} "
                    "Cadastre um documento orientativo para que o sistema passe a instruir a correcao."
                    + _aviso_encaixe(rpm, similaridade.regime_rpm, aproximado)
                ),
                **base_diagnostico,
            )

        trechos = self.base.buscar(
            f"{rotulo} {entrada['termos_busca']}",
            top_k=self.top_k,
            documento=cobertura.documento,
            preferir_secoes=SECOES_DE_ACAO,
        )
        instrucoes, gerador = self._gerar_com_fallback(
            INSTRUCAO, montar_pergunta(rotulo, estatistica, trechos)
        )

        return Diagnostico(
            situacao="defeito_documentado",
            mensagem=(
                f"O padrao identificado e {rotulo}, com {_frase_historico(estatistica)}. "
                f"{cobertura.motivo}"
                + _aviso_encaixe(rpm, similaridade.regime_rpm, aproximado)
            ),
            trechos=[asdict(t) for t in trechos],
            instrucoes=instrucoes,
            gerador=gerador,
            **base_diagnostico,
        )

    def perguntar(self, pergunta: str, *, documento: str | None = None) -> dict[str, Any]:
        """Consulta livre a base documental, usada pelo chat.

        Passa pelo mesmo portao de cobertura do diagnostico, por tres criterios:

        1. cobertura - se a pergunta nomeia um defeito do catalogo que nenhum
           procedimento cobre, a resposta e a mesma recusa do diagnostico, com
           pedido de cadastro, e o modelo nao e chamado. Se o defeito nomeado e
           coberto, o contexto fica restrito ao documento que o cobre, como no
           caminho do diagnostico;
        2. aderencia - o melhor trecho precisa passar de PISO_SCORE_CHAT. O
           filtro anterior era "score > 0", que aceitava qualquer coincidencia
           de palavra;
        3. concentracao - os trechos precisam vir do MESMO documento. Resposta
           montada com pedaco de tres procedimentos diferentes e o sintoma de
           que nenhum deles trata do assunto.

        Os criterios 2 e 3 valem para a pergunta que nao nomeia defeito nenhum:
        e ela que antes entrava sem portao algum.
        """
        citados = self._defeitos_citados(pergunta)
        sem_documento = [rotulo for rotulo, cobertura in citados if not cobertura.coberto]
        if sem_documento:
            return _recusa_chat(
                "defeito_sem_documentacao",
                f'Nenhum procedimento cadastrado trata de "{"; ".join(sem_documento)}". '
                "Cadastre um documento orientativo sobre esse defeito para que o sistema "
                "passe a instruir a correcao.",
            )

        cobertos = sorted((c for _, c in citados if c.coberto), key=lambda c: c.score, reverse=True)
        do_defeito = cobertos[0].documento if cobertos else None
        trechos = self.base.buscar(pergunta, top_k=self.top_k, documento=documento or do_defeito)
        if not trechos:
            return _recusa_chat("fora_de_escopo", SEM_ASSUNTO)

        if do_defeito is None:
            if trechos[0].score < PISO_SCORE_CHAT:
                return _recusa_chat("fora_de_escopo", SEM_ASSUNTO)
            documentos = sorted({t.documento for t in trechos})
            if len(documentos) > 1:
                return _recusa_chat(
                    "fora_de_escopo",
                    "Nenhum procedimento cadastrado trata desse assunto por inteiro: os "
                    f"trechos mais aderentes ficaram espalhados por {len(documentos)} "
                    f"documentos ({', '.join(documentos)}), sinal de que nenhum deles cobre "
                    "a pergunta. Reformule citando o defeito, ou escolha um documento.",
                )

        contexto = "\n---\n".join(f"({t.documento} / {t.secao})\n{t.texto}" for t in trechos)
        resposta, gerador = self._gerar_com_fallback(
            INSTRUCAO_CHAT,
            f"PERGUNTA: {pergunta}\n\nTRECHOS DO PROCEDIMENTO:\n{contexto}",
        )
        return {
            "situacao": "respondido",
            "documento": trechos[0].documento,
            "resposta": resposta,
            "trechos": [asdict(t) for t in trechos],
            "gerador": gerador,
        }

    def _defeitos_citados(self, pergunta: str) -> list[tuple[str, Cobertura]]:
        """Defeitos do catalogo que a pergunta nomeia, com a cobertura de cada um.

        O casamento exige todos os radicais do termo-chave dentro da pergunta -
        a mesma exigencia que a cobertura faz contra o escopo do documento. Por
        isso "rotor excentrico" nao casa com o catalogo do rotor inclinado, que
        pede os radicais de "rotor" e de "inclinado".
        """
        radicais = stems(pergunta)
        return [
            (
                entrada["rotulo"],
                self.base.cobertura(
                    entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
                ),
            )
            for entrada in self.catalogo.values()
            if stems(entrada["termo_chave"]) <= radicais
        ]

    def _gerar_com_fallback(self, instrucao: str, pergunta: str) -> tuple[str, str]:
        """Gera o texto e devolve tambem qual gerador o produziu.

        O fallback do factory e avaliado uma unica vez, no carregamento, e o
        motor vive o processo inteiro. Sem este degrau, um Ollama que cai depois
        do boot derruba um diagnostico que ja estava pronto - defeito
        identificado, historico, cobertura e trechos recuperados - e devolve 500.
        O nome sai marcado para que o tecnico saiba que o texto veio recortado do
        procedimento em vez de reescrito, em vez de a troca ser silenciosa.
        """
        try:
            return self.gerador.gerar(instrucao, pergunta), self.gerador.nome
        except Exception:  # noqa: BLE001 - qualquer falha do servico externo cai para o recorte
            LOGGER.exception("gerador %s falhou; recortando o procedimento", self.gerador.nome)
            reserva = GeradorDeterministico()
            return reserva.gerar(instrucao, pergunta), f"{reserva.nome} (fallback em execucao)"


def _recusa_chat(situacao: str, mensagem: str) -> dict[str, Any]:
    """Resposta do chat sem chamar o modelo. Sem trecho, nao ha o que reescrever."""
    return {
        "situacao": situacao,
        "documento": None,
        "resposta": mensagem,
        "trechos": [],
        "gerador": "-",
    }


def _rotacao_sem_historico(rpm: float) -> Diagnostico:
    conhecidos = ", ".join(f"{int(v)}" for v in RPM_CONHECIDOS)
    return Diagnostico(
        situacao="padrao_desconhecido",
        fault=None,
        rotulo=None,
        confianca=0.0,
        regime_rpm=REGIME_FORA_DA_GRADE,
        e_defeito=False,
        mensagem=(
            f"A rotacao de {rpm:.0f} rpm nao tem historico comparavel. O ensaio operou em "
            f"{conhecidos} rpm e a vibracao escala com a rotacao, entao comparar este evento "
            "com outro regime produziria um diagnostico sem lastro. Registre a condicao "
            "observada nesta rotacao para que ela passe a fazer parte da base."
        ),
        rpm=rpm,
    )


def _aviso_encaixe(rpm: float, regime: str, aproximado: bool) -> str:
    if not aproximado:
        return ""
    return (
        f" Atencao: a rotacao medida foi de {rpm:.0f} rpm e a comparacao usou o historico de "
        f"{regime}, o regime conhecido mais proximo dentro da tolerancia."
    )


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
        # A contagem crua acompanha o peso porque e ela que o portao de consenso
        # usa e a unica das duas que um tecnico consegue interpretar: "18 dos 25
        # vizinhos" diz mais do que "74% do peso".
        "votos": similaridade.votos,
        "vizinhos_consultados": len(similaridade.vizinhos),
        "consenso_simples": similaridade.consenso_simples,
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


def montar_pergunta(
    rotulo: str, estatistica: EstatisticaHistorica, trechos: list[TrechoRecuperado]
) -> str:
    contexto = "\n---\n".join(f"({t.documento} / {t.secao})\n{t.texto}" for t in trechos)
    return (
        f"DEFEITO IDENTIFICADO: {rotulo}\n"
        f"HISTORICO: {_frase_historico(estatistica)}.\n\n"
        f"TRECHOS DO PROCEDIMENTO:\n{contexto}"
    )
