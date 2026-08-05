"""O chat obedece a mesma regra do diagnostico: nao fala sem documento.

O enunciado e literal - "o sistema deve se deter unicamente a problemas que
possuem documentos". Antes destes testes a regra valia so no caminho do
diagnostico: perguntar "como corrigir rotor excentrico?" no chat devolvia
trechos do procedimento de rotor INCLINADO e o modelo escrevia a correcao, para
o defeito que o README anuncia como sem procedimento cadastrado.

O gerador aqui e um duble que registra o que recebeu. Nenhum teste depende do
Ollama estar no ar, e todos falham se o portao deixar a chamada acontecer.
"""

from __future__ import annotations

import pytest

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.knowledge.store import BaseConhecimento
from prescritiva.similarity.index import IndiceSimilaridade


class GeradorEspiao:
    """Registra as chamadas em vez de gerar. Se for chamado, o portao vazou."""

    nome = "espiao"

    def __init__(self) -> None:
        self.chamadas: list[tuple[str, str]] = []

    def disponivel(self) -> bool:
        return True

    def gerar(self, instrucao: str, pergunta: str) -> str:
        self.chamadas.append((instrucao, pergunta))
        return "resposta do duble"


@pytest.fixture(scope="module")
def espiao() -> GeradorEspiao:
    return GeradorEspiao()


@pytest.fixture(scope="module")
def motor(espiao: GeradorEspiao) -> MotorDiagnostico:
    settings = load_settings()
    if not (settings.paths.index_dir / "base_conhecimento.json").exists():
        pytest.skip("execute scripts/build_knowledge.py e build_index.py antes")
    return MotorDiagnostico(
        indice=IndiceSimilaridade.carregar(settings.paths.index_dir),
        base=BaseConhecimento.carregar(settings.paths.index_dir),
        database=settings.paths.database,
        gerador=espiao,
        top_k=settings.knowledge["retrieval_top_k"],
    )


@pytest.mark.parametrize(
    "pergunta",
    [
        "como corrigir rotor excentrico?",
        "qual o procedimento para excentricidade do rotor?",
        "como consertar a ventoinha do motor?",
        "a ventoinha esta fazendo barulho, o que fazer?",
    ],
)
def test_chat_recusa_defeito_sem_procedimento(motor, espiao, pergunta):
    """Os dois defeitos que o enunciado quer ver recusados, agora pelo chat."""
    antes = len(espiao.chamadas)
    resultado = motor.perguntar(pergunta)

    assert resultado["situacao"] == "defeito_sem_documentacao"
    assert not resultado["trechos"]
    assert "nenhum procedimento" in resultado["resposta"].lower()
    assert "cadastre" in resultado["resposta"].lower()
    # O modelo nem chega a ser chamado: nao existe texto aprovado para reescrever.
    assert len(espiao.chamadas) == antes


@pytest.mark.parametrize(
    "pergunta",
    [
        "como trocar o oleo do motor a diesel?",
        "como calibrar o sensor de pressao da caldeira a vapor?",
        "quais os requisitos da norma de seguranca eletrica NR10?",
    ],
)
def test_chat_recusa_assunto_fora_dos_procedimentos(motor, espiao, pergunta):
    """Sem defeito nomeado nao ha o que ancorar, entao nao ha o que prescrever.

    A recusa mudou de nome junto com o portao. Antes era "fora_de_escopo", que
    descrevia o resultado de um piso de aderencia; hoje "sem_defeito_nomeado"
    descreve a causa, e "fora_de_escopo" ficou reservado para o caso em que o
    defeito foi ancorado e ainda assim nao sobrou trecho.
    """
    antes = len(espiao.chamadas)
    resultado = motor.perguntar(pergunta)

    assert resultado["situacao"] == "sem_defeito_nomeado"
    assert not resultado["trechos"]
    assert len(espiao.chamadas) == antes
    # Recusa que ensina: diz o que falta na pergunta e o que o sistema cobre.
    assert "nao nomeia nenhum defeito" in resultado["resposta"].lower()
    assert "ha procedimento para" in resultado["resposta"].lower()


def test_chat_responde_defeito_documentado_com_um_unico_documento(motor, espiao):
    resultado = motor.perguntar("como corrigir desalinhamento de eixo?")

    assert resultado["situacao"] == "respondido"
    assert resultado["documento"] == "Doc2"
    assert resultado["trechos"]
    # O contexto entregue ao modelo nao pode misturar procedimentos: uma resposta
    # montada com pedaco de tres documentos e uma resposta sem procedimento.
    assert {t["documento"] for t in resultado["trechos"]} == {"Doc2"}
    assert espiao.chamadas


@pytest.mark.parametrize(
    ("pergunta", "documento"),
    [
        ("como corrigir desalinhamento angular?", "Doc2"),
        ("como substituir a correia?", "Doc4"),
        ("como instalar um rolamento novo?", "Doc1"),
        ("como alinhar a polia?", "Doc5"),
        ("como corrigir rotor inclinado?", "Doc6"),
        ("como balancear o rotor desbalanceado?", "Doc3"),
    ],
)
def test_chat_continua_respondendo_os_seis_procedimentos(motor, pergunta, documento):
    """Contraprova do portao: um filtro que recusa tudo tambem estaria errado.

    Cada um dos seis procedimentos cadastrados precisa continuar respondendo, e
    pelo documento certo - senao a correcao teria trocado alucinacao por mudez.
    """
    resultado = motor.perguntar(pergunta)

    assert resultado["situacao"] == "respondido"
    assert resultado["documento"] == documento
    assert {t["documento"] for t in resultado["trechos"]} == {documento}


def test_chat_usa_instrucao_propria_sem_diagnostico_pressuposto(motor, espiao):
    espiao.chamadas.clear()
    motor.perguntar("como tensionar a correia?")

    instrucao, _ = espiao.chamadas[-1]
    # A INSTRUCAO do diagnostico afirma que "o tipo de defeito ja foi
    # determinado pela analise de similaridade". No chat nao foi determinado
    # nada, e a frase autoriza o modelo a supor um defeito.
    assert "ja foi determinado" not in instrucao
    assert "nao levante hipotese" in instrucao.lower().replace("\n", " ")


def test_chat_com_documento_escolhido_ainda_passa_pelo_portao(motor, espiao):
    """Escolher o documento na tela nao pode contornar a regra de cobertura."""
    antes = len(espiao.chamadas)
    resultado = motor.perguntar("como corrigir rotor excentrico?", documento="Doc6")

    assert resultado["situacao"] == "defeito_sem_documentacao"
    assert len(espiao.chamadas) == antes


# ---------------------------------------------------------------------------
# Os tres contornos que a auditoria adversarial encontrou.
#
# O portao anterior so barrava a frase literal. scripts/auditoria_chat.py mediu
# 66 vazamentos em 101 perguntas: bastava parafrasear, ou citar junto um defeito
# coberto, ou fixar um documento no combo da tela. As perguntas abaixo sao
# amostras daquele corpus, guardadas aqui para que o buraco nao volte.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pergunta",
    [
        "O rotor esta fora de centro em relacao ao estator. Como corrigir?",
        "Como corrigir um rotor descentralizado dentro do estator?",
        "O rotor gira fora de eixo e a corrente oscila a cada volta. Qual o procedimento?",
        "O ar de refrigeracao caiu e o motor esquenta, tem sujeira acumulada nas pas. Como corrijo?",
        "A folga entre a parte que gira e a carcaca ta maior de um lado que do outro.",
        "O motor faz um ronco que vai e volta, tipo pulsando, e a trepidacao acompanha a rotacao.",
    ],
)
def test_chat_nao_prescreve_para_parafrase_de_defeito_sem_procedimento(motor, espiao, pergunta):
    """Parafrase nao pode virar prescricao.

    O corpo dos procedimentos nao distingue estes casos: o Doc5 tem uma secao
    "3.1 Excentricidade" sobre excentricidade DA POLIA, e uma pergunta sobre
    excentricidade do ROTOR casa com ela quase palavra por palavra. Por isso o
    portao nao pode ser um limiar de aderencia - foi medido que os scores de
    pergunta legitima e de parafrase se sobrepoem inteiros.
    """
    antes = len(espiao.chamadas)
    resultado = motor.perguntar(pergunta)

    assert resultado["situacao"] != "respondido"
    assert not resultado["trechos"]
    assert len(espiao.chamadas) == antes


def test_chat_recusa_quando_a_tela_restringe_a_procedimento_incompativel(motor, espiao):
    """O combo da tela ESTREITA o que o catalogo aprovou; nao cria aprovacao.

    Antes o parametro `documento` entrava direto na busca e, de quebra, fazia a
    checagem de "trechos do mesmo documento" passar por construcao - filtrar a um
    documento so deixa um documento so. Era o contorno mais facil de todos.
    """
    antes = len(espiao.chamadas)
    resultado = motor.perguntar("como tensionar a correia?", documento="Doc1")

    assert resultado["situacao"] == "restricao_incompativel"
    assert not resultado["trechos"]
    assert len(espiao.chamadas) == antes
    # A recusa precisa dizer para onde ir, nao so que nao pode.
    assert "Doc4" in resultado["resposta"]


def test_chat_aceita_restricao_compativel(motor):
    """Contraprova: restringir ao procedimento certo continua funcionando."""
    resultado = motor.perguntar("como tensionar a correia?", documento="Doc4")

    assert resultado["situacao"] == "respondido"
    assert resultado["documento"] == "Doc4"


def test_chat_com_dois_defeitos_citados_deixa_a_aderencia_escolher(motor):
    """Quando a pergunta nomeia dois defeitos cobertos, ambos sao candidatos.

    O catalogo decide QUAIS procedimentos entram; a aderencia do texto decide
    QUAL deles responde. Restringir a um antes de olhar a pergunta seria um
    palpite tomado cedo demais.
    """
    resultado = motor.perguntar("a correia e a polia estao alinhadas? como verifico o canal da polia?")

    assert resultado["situacao"] == "respondido"
    assert resultado["documento"] in {"Doc4", "Doc5"}


def test_resposta_declara_de_que_defeito_e_o_procedimento(motor):
    """A resposta carrega o defeito do procedimento que ela citou.

    E o que separa "instrucao sem origem" de "instrucao com etiqueta". Quem
    pergunta citando um defeito coberto recebe o procedimento DELE, e ve isso
    escrito - inclusive quando o que o tecnico queria era outra coisa.
    """
    resultado = motor.perguntar("como corrigir desalinhamento de eixo?")

    assert resultado["situacao"] == "respondido"
    assert resultado["defeito"] == "Desalinhamento de eixo"
    assert resultado["documento"] == "Doc2"


def test_toda_recusa_traz_os_mesmos_campos_da_resposta(motor):
    """Contrato estavel: quem consome nao precisa saber qual recusa aconteceu."""
    chaves = {"situacao", "documento", "defeito", "resposta", "trechos", "gerador"}
    for pergunta, documento in [
        ("como corrigir rotor excentrico?", None),
        ("como trocar o oleo do motor a diesel?", None),
        ("como tensionar a correia?", "Doc1"),
        ("como corrigir desalinhamento de eixo?", None),
    ]:
        assert set(motor.perguntar(pergunta, documento=documento)) == chaves
