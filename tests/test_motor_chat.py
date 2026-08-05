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
    antes = len(espiao.chamadas)
    resultado = motor.perguntar(pergunta)

    assert resultado["situacao"] == "fora_de_escopo"
    assert not resultado["trechos"]
    assert "nenhum procedimento" in resultado["resposta"].lower()
    assert len(espiao.chamadas) == antes


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
