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

# Nao existe piso de aderencia na consulta livre, e a ausencia e deliberada.
# Houve um: PISO_SCORE_CHAT = 2.5, aplicado quando a pergunta nao nomeava defeito.
# A auditoria adversarial mediu que aquele ramo nao tinha poder de decisao - as
# distribuicoes de score da pergunta legitima e da parafrase de defeito sem
# procedimento se sobrepoem inteiras -, e qualquer piso ali troca vazamento por
# mudez quase na proporcao de um para um. Substituir um numero ruim por um numero
# melhor manteria o erro de metodo; o portao passou a ser o catalogo, que decide
# por escopo. Ver perguntar() e scripts/auditoria_chat.py.

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

        trechos, pergunta = self._recuperar_trechos_e_pergunta(fault, cobertura.documento, estatistica)
        instrucoes, gerador = self._gerar_com_fallback(INSTRUCAO, pergunta)

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

    def _recuperar_trechos_e_pergunta(
        self, fault: str, documento: str, estatistica: EstatisticaHistorica
    ) -> tuple[list[TrechoRecuperado], str]:
        """Recupera os trechos do procedimento e monta o prompt para um defeito ja coberto.

        Unico ponto que decide COMO buscar (quais termos, que documento, quais
        secoes preferir) para gerar uma prescricao. `_defeito()` usa isto para
        montar o diagnostico; `preparar_prescricao()` expoe o mesmo caminho para
        quem precisa regenerar a partir de um diagnostico ja aprovado, como
        `scripts/comparar_modelos.py` - um portao novo que entre aqui vale para
        os dois sem precisar lembrar de replicar a busca em cada lugar que gera
        texto.
        """
        entrada = self.catalogo[fault]
        trechos = self.base.buscar(
            f"{entrada['rotulo']} {entrada['termos_busca']}",
            top_k=self.top_k,
            documento=documento,
            preferir_secoes=SECOES_DE_ACAO,
        )
        return trechos, montar_pergunta(entrada["rotulo"], estatistica, trechos)

    def preparar_prescricao(self, diagnostico: Diagnostico) -> tuple[list[TrechoRecuperado], str]:
        """Trechos e prompt para um diagnostico ja aprovado pelos tres portoes.

        Exige `situacao == "defeito_documentado"`: os outros tres desfechos nao
        tem procedimento para recuperar, e chamar aqui para eles e erro de quem
        chama, nao caso a tratar. `_defeito()` chega ate aqui internamente para
        montar a prescricao original; este metodo existe para quem quer refazer
        a busca e a geracao depois - hoje so `scripts/comparar_modelos.py`, para
        comparar geradores no mesmo caso real sem reconstruir a consulta por
        conta propria.
        """
        if diagnostico.situacao != "defeito_documentado" or diagnostico.fault is None:
            raise ValueError(
                "preparar_prescricao exige um diagnostico defeito_documentado, "
                f"recebeu situacao={diagnostico.situacao!r}"
            )
        estatistica = estatisticas(self.database, diagnostico.fault)
        return self._recuperar_trechos_e_pergunta(
            diagnostico.fault, diagnostico.cobertura["documento"], estatistica
        )

    def perguntar(self, pergunta: str, *, documento: str | None = None) -> dict[str, Any]:
        """Consulta livre a base documental, usada pelo chat.

        A regra e uma so, e vale igual para as tres formas de perguntar:

            o CATALOGO decide se responde e quais procedimentos sao candidatos;
            a ADERENCIA do texto decide qual candidato - nunca se responde.

        A versao anterior invertia essa ordem quando a pergunta nao nomeava
        defeito nenhum: caia num BM25 sobre o corpo inteiro, com um piso de score
        e uma exigencia de documento unico. Foi medido que esse ramo nao decide
        nada. Sobre 101 perguntas adversariais e 30 legitimas, a geometria do
        score e a mesma nos dois grupos - mediana de aderencia 6,88 contra 7,83,
        margem para o segundo documento 1,72 contra 1,62 - e nenhum limiar separa
        os dois sem emudecer o produto. scripts/auditoria_chat.py refaz a conta.

        A causa e visivel no acervo. O Doc5 tem uma secao inteira chamada
        "3.1 Excentricidade", que abre com "ocorre quando o centro geometrico da
        POLIA nao coincide com o centro de rotacao". Uma pergunta sobre
        excentricidade do ROTOR - defeito que nenhum procedimento cobre - casa
        quase palavra por palavra com esse trecho. O corpo nao distingue o
        defeito coberto do defeito homonimo em outro componente; so o escopo
        distingue, e o escopo e consultado pelo catalogo.

        Dai as tres saidas sem chamar o modelo:

        1. a pergunta nomeia um defeito que nenhum procedimento cobre - recusa e
           pede o cadastro, como no caminho do diagnostico;
        2. a pergunta nao nomeia defeito algum - nao ha o que ancorar, e a
           resposta diz sobre o que o sistema sabe falar;
        3. a tela restringiu a um procedimento que nao esta entre os candidatos -
           o combo da interface pode ESTREITAR o que o catalogo aprovou, nunca
           criar aprovacao.
        """
        ancoradas = self._defeitos_citados(pergunta)

        sem_documento = [rotulo for rotulo, cobertura in ancoradas if not cobertura.coberto]
        if sem_documento:
            return _recusa_chat(
                "defeito_sem_documentacao",
                f'Nenhum procedimento cadastrado trata de "{"; ".join(sem_documento)}". '
                "Cadastre um documento orientativo sobre esse defeito para que o sistema "
                "passe a instruir a correcao.",
            )

        pendente = self._fenomeno_sem_dono(pergunta)
        if pendente is not None:
            termo, rotulo, donos = pendente
            return _recusa_chat(
                "fenomeno_ambiguo",
                f'"{termo.capitalize()}" tanto pode ser {rotulo}, que nao tem procedimento '
                f"cadastrado, quanto o mesmo fenomeno em {donos}, que tem. A pergunta nao "
                "diz qual dos dois, e prescrever pelo palpite entregaria ao tecnico o "
                f"procedimento do componente errado. Diga em que componente, ou cadastre um "
                f"documento orientativo sobre {rotulo.lower()}.",
            )

        cobertas = [(rotulo, c) for rotulo, c in ancoradas if c.coberto]
        if not cobertas:
            return _recusa_chat("sem_defeito_nomeado", self._sobre_o_que_sei_falar())

        # Um documento pode ser candidato por mais de um defeito (os quatro modos
        # de falha de rolamento caem todos no Doc1), e uma pergunta pode nomear
        # dois defeitos de documentos diferentes ("a correia e a polia"). O mapa
        # guarda o rotulo para que a resposta possa declarar de que defeito e o
        # procedimento que ela esta citando.
        candidatos: dict[str, str] = {}
        for rotulo, cobertura in sorted(cobertas, key=lambda item: item[1].score, reverse=True):
            candidatos.setdefault(cobertura.documento, rotulo)

        if documento is not None and documento not in candidatos:
            return _recusa_chat(
                "restricao_incompativel",
                f"A pergunta foi restrita a {documento}, mas esse procedimento nao trata do "
                f"que foi perguntado. Os procedimentos que tratam sao: "
                f"{', '.join(f'{doc} ({rot})' for doc, rot in sorted(candidatos.items()))}. "
                "Escolha um deles ou remova a restricao.",
            )

        permitidos = {documento} if documento is not None else set(candidatos)
        trechos = self.base.buscar(pergunta, top_k=self.top_k, documento=permitidos)
        if not trechos:
            # O portao e a recuperacao normalizam texto de formas diferentes: o
            # portao compara RADICAIS e o BM25 compara TOKENS. Por isso
            # "como balanceio um ventilador em campo?" ancora no Doc3 pelo radical
            # "balanc" e depois nao recupera nada dele - o acervo escreve
            # "balanceamento", nunca "balanceio". Com o defeito ja aprovado, cair
            # fora seria recusar por vocabulario, nao por cobertura. Entao a
            # segunda tentativa usa o vocabulario canonico do proprio catalogo,
            # que e o mesmo que o caminho do diagnostico usa.
            canonico = " ".join(
                f"{rotulo} {self._termos_do_rotulo(rotulo)}"
                for doc, rotulo in candidatos.items()
                if doc in permitidos
            )
            trechos = self.base.buscar(canonico, top_k=self.top_k, documento=permitidos)
        if not trechos:
            return _recusa_chat("fora_de_escopo", SEM_ASSUNTO)

        escolhido = trechos[0].documento
        defeito = candidatos[escolhido]
        contexto = "\n---\n".join(f"({t.documento} / {t.secao})\n{t.texto}" for t in trechos)
        resposta, gerador = self._gerar_com_fallback(
            INSTRUCAO_CHAT,
            f"PROCEDIMENTO CONSULTADO: {escolhido}, que trata de {defeito}.\n"
            f"PERGUNTA: {pergunta}\n\nTRECHOS DO PROCEDIMENTO:\n{contexto}",
        )
        return {
            "situacao": "respondido",
            "documento": escolhido,
            # O defeito viaja com a resposta para que a tela e a API possam dizer
            # de que procedimento o texto saiu. Quem pergunta "ja troquei o
            # rolamento, e agora?" recebe, no maximo, o procedimento de rolamento
            # com essa etiqueta na frente, em vez de uma instrucao sem origem.
            "defeito": defeito,
            "resposta": resposta,
            "trechos": [asdict(t) for t in trechos],
            "gerador": gerador,
        }

    def _termos_do_rotulo(self, rotulo: str) -> str:
        """Vocabulario de busca que o catalogo declara para este defeito."""
        for entrada in self.catalogo.values():
            if entrada["rotulo"] == rotulo:
                return entrada["termos_busca"]
        return rotulo

    def _fenomeno_sem_dono(self, pergunta: str) -> tuple[str, str, str] | None:
        """Fenomeno citado que pode ser de um defeito sem procedimento.

        Dois fenomenos deste acervo pertencem a mais de um defeito, e o que
        decide de qual e o COMPONENTE, nunca a palavra:

            excentricidade  do rotor nao tem procedimento; da polia tem, e o
                            Doc5 dedica a ela a secao "3.1 Excentricidade";
            ventilador      falha da propria ventoinha nao tem procedimento;
                            desbalanceamento de ventilador tem, porque o escopo
                            do Doc3 declara cobrir ventiladores.

        Citar o fenomeno sem citar o componente deixa a pergunta indecidivel. O
        portao anterior resolvia isso pelo BM25, que escolhia o documento com
        mais palavras em comum - e foi medido que essa escolha nao tem lastro.
        Aqui a indecisao vira pergunta de volta ao tecnico.

        O termo so bloqueia quando o dono legitimo NAO foi nomeado. E o que
        separa esta regra de uma lista de palavras proibidas: "como corrigir a
        excentricidade da polia?" continua respondendo pelo Doc5.
        """
        radicais = stems(pergunta)
        nomeados = {
            fault
            for fault, entrada in self.catalogo.items()
            if any(stems(forma) <= radicais for forma in _formas_de_nomear(entrada))
        }
        for fault, entrada in self.catalogo.items():
            if fault in nomeados:
                continue
            cobertura = self.base.cobertura(
                entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
            )
            if cobertura.coberto:
                continue
            for ambiguo in entrada.get("termos_ambiguos", []):
                if not stems(ambiguo["termo"]) <= radicais:
                    continue
                donos = [d for d in ambiguo["resolvido_por"] if d not in nomeados]
                if len(donos) == len(ambiguo["resolvido_por"]):
                    rotulos = ", ".join(
                        self.catalogo[d]["rotulo"].lower() for d in ambiguo["resolvido_por"]
                    )
                    return ambiguo["termo"], entrada["rotulo"], rotulos
        return None

    def _sobre_o_que_sei_falar(self) -> str:
        """Recusa que ensina, em vez de recusa que so fecha a porta.

        Sem esta lista o tecnico recebe "reformule" e nao tem como saber para
        onde reformular. A lista sai da cobertura medida na hora, nao de texto
        fixo: cadastrar um procedimento novo muda a resposta sozinho.
        """
        tratados = sorted(
            {
                f"{cobertura.documento} ({entrada['rotulo']})"
                for entrada in self.catalogo.values()
                for cobertura in [
                    self.base.cobertura(
                        entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
                    )
                ]
                if cobertura.coberto
            }
        )
        return (
            "A pergunta nao nomeia nenhum defeito com procedimento cadastrado, e sem isso "
            "nao ha procedimento de onde tirar a instrucao. Diga qual defeito voce suspeita, "
            "ou cadastre um documento orientativo sobre ele. "
            f"Hoje ha procedimento para: {'; '.join(tratados)}."
        )

    def _defeitos_citados(self, pergunta: str) -> list[tuple[str, Cobertura]]:
        """Defeitos do catalogo que a pergunta nomeia, com a cobertura de cada um.

        Cada defeito declara uma ou mais formas de ser nomeado, e basta UMA
        casar. O casamento de cada forma continua exigindo todos os radicais
        dentro da pergunta - a mesma exigencia que a cobertura faz contra o
        escopo do documento -, entao "rotor excentrico" nao casa com o catalogo
        do rotor inclinado, que pede os radicais de "rotor" e de "inclinado".

        A lista de formas existe porque o radical e o corte em seis letras
        separam palavras que o tecnico usa como sinonimo: "balanceamento" vira
        "balanc" e "desbalanceamento" vira "desbal"; "cocked rotor" e o nome que
        o proprio Doc6 usa no titulo e nao casa com "rotor inclinado". Sem elas o
        portao recusava pergunta legitima, que e a falha pior.

        tests/test_catalogo_vocabulario.py impede que essa lista vire palpite:
        forma de defeito COBERTO tem que existir no texto do documento que o
        cobre, e forma de defeito SEM procedimento nao pode aparecer no escopo de
        documento nenhum.
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
            if any(stems(forma) <= radicais for forma in _formas_de_nomear(entrada))
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


def _formas_de_nomear(entrada: dict[str, Any]) -> list[str]:
    """Formas pelas quais uma pergunta pode nomear este defeito.

    O termo-chave sempre entra: ele e o que decide a cobertura contra o escopo do
    documento, entao nomear o defeito por ele nunca pode falhar. `termos_pergunta`
    e opcional e so acrescenta.
    """
    return [entrada["termo_chave"], *entrada.get("termos_pergunta", [])]


def _recusa_chat(situacao: str, mensagem: str) -> dict[str, Any]:
    """Resposta do chat sem chamar o modelo. Sem trecho, nao ha o que reescrever."""
    return {
        "situacao": situacao,
        "documento": None,
        "defeito": None,
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
