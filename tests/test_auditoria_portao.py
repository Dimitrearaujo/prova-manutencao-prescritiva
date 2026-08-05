"""Trava os numeros da auditoria adversarial do chat.

scripts/auditoria_chat.py mede; este arquivo impede que a medida regrida sem
alguem perceber. A diferenca importa: o numero que so existe num README envelhece
em silencio, e este projeto ja perdeu credibilidade uma vez por afirmar sem medir.

Procedencia do corpus (tests/dados/corpus_chat_adversarial.json): perguntas
geradas por tres agentes adversariais com angulos distintos - sintoma, sinonimo e
contorno estrutural - e depois CURADAS por revisores que julgaram cada uma contra
o texto real dos seis procedimentos. A curadoria nao foi cosmetica: de 101
perguntas geradas como ataque, 22 descreviam defeito legitimamente coberto e
viraram controle, e 13 ficaram ambiguas a ponto de nenhum engenheiro decidir sem
espectro ou medicao, e foram excluidas dos dois denominadores. Reportar contra o
corpus bruto teria errado nos dois sentidos.

O corpus NAO foi usado para escolher o vocabulario do catalogo. Ele apontou ONDE
havia buraco; quem decidiu QUAL termo o fecha foi o texto dos documentos, sob a
regra de tests/test_catalogo_vocabulario.py. Ajustar o catalogo ate o corpus
ficar verde seria repetir o erro do conserto anterior, que cobriu exatamente as
frases da auditoria e mais nenhuma.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.knowledge.store import BaseConhecimento
from prescritiva.similarity.index import IndiceSimilaridade

CORPUS = Path(__file__).parent / "dados" / "corpus_chat_adversarial.json"

# Tetos medidos em 2026-08-05, com o portao ja corrigido. Sao trava de catraca:
# podem descer quando o portao melhorar, nunca subir sem decisao explicita.
TETO_COCITACAO = 16
TETO_MUDEZ_BEM_FORMADA = 1


class GeradorEspiao:
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


@pytest.fixture(scope="module")
def corpus() -> dict:
    if not CORPUS.exists():
        pytest.skip(f"corpus ausente: {CORPUS}")
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def _vazamentos(motor: MotorDiagnostico, casos: list[dict]) -> list[dict]:
    vazou = []
    for caso in casos:
        saida = motor.perguntar(
            caso["pergunta"], documento=caso.get("documento_seletor") or None
        )
        if saida["situacao"] == "respondido":
            vazou.append({**caso, "respondeu_por": saida["documento"]})
    return vazou


def test_o_corpus_tem_massa_para_medir(corpus):
    """Guarda contra um corpus esvaziado deixar todos os testes verdes."""
    por_mecanismo = {m: 0 for m in ("parafrase", "cocitacao", "seletor")}
    for caso in corpus["ataques"]:
        por_mecanismo[caso["mecanismo"]] = por_mecanismo.get(caso["mecanismo"], 0) + 1
    assert por_mecanismo["parafrase"] >= 25
    assert por_mecanismo["cocitacao"] >= 15
    assert por_mecanismo["seletor"] >= 8
    assert len(corpus["controles"]) >= 40


def test_parafrase_nao_vaza(motor, corpus):
    """Invariante duro: nenhuma parafrase pode virar prescricao.

    Era o buraco central. O conserto anterior fechou as frases literais da
    auditoria e deixou este mecanismo inteiro aberto.
    """
    casos = [c for c in corpus["ataques"] if c["mecanismo"] == "parafrase"]
    vazou = _vazamentos(motor, casos)
    assert not vazou, "\n".join(f"{v['pergunta']} -> {v['respondeu_por']}" for v in vazou)


def test_seletor_da_tela_nao_vaza(motor, corpus):
    """Invariante duro: fixar o documento na tela nao pode criar aprovacao."""
    casos = [c for c in corpus["ataques"] if c["mecanismo"] == "seletor"]
    vazou = _vazamentos(motor, casos)
    assert not vazou, "\n".join(f"{v['pergunta']} -> {v['respondeu_por']}" for v in vazou)


def test_cocitacao_nao_piora(motor, corpus):
    """Residuo conhecido, declarado e travado por catraca.

    Quando a pergunta nomeia um defeito COBERTO e descreve, sem nomear, um
    defeito sem procedimento - "ja troquei o rolamento e o ar continua fraco" -,
    o sistema responde pelo procedimento do defeito nomeado. Ele nao inventa
    texto e nao prescreve para o defeito sem documento; entrega o procedimento de
    outro defeito, com a etiqueta dizendo de qual defeito ele e.

    Nao foi consertado de proposito. O conserto exigiria detectar que o defeito
    citado foi DESCARTADO pelo tecnico, e as formas de escrever isso em portugues
    nao cabem numa lista sem virar o mesmo jogo de adivinhacao que este portao
    saiu de dentro.
    """
    casos = [c for c in corpus["ataques"] if c["mecanismo"] == "cocitacao"]
    vazou = _vazamentos(motor, casos)
    assert len(vazou) <= TETO_COCITACAO, (
        f"co-citacao regrediu: {len(vazou)} vazamentos, teto {TETO_COCITACAO}\n"
        + "\n".join(f"{v['pergunta']} -> {v['respondeu_por']}" for v in vazou)
    )


def test_pergunta_bem_formada_continua_respondida(motor, corpus):
    """Contraprova obrigatoria: um portao que recusa tudo tambem zera vazamento.

    Só entram aqui as perguntas de manutencao bem formadas, que nomeiam o defeito.
    As reclassificadas da geracao adversarial descrevem sintoma sem nomear defeito
    e sao medidas a parte, em scripts/auditoria_chat.py - o chat nao tem sensor e
    responder sintoma por semelhanca de palavra e exatamente o que foi medido nao
    funcionar.
    """
    bem_formadas = [c for c in corpus["controles"] if not c.get("origem")]
    assert len(bem_formadas) >= 25, "o conjunto de controle encolheu"

    perdidas = []
    for caso in bem_formadas:
        saida = motor.perguntar(caso["pergunta"])
        if saida["situacao"] != "respondido":
            perdidas.append(f"{caso['pergunta']} -> recusou ({saida['situacao']})")
        elif saida["documento"] != caso["documento_esperado"]:
            perdidas.append(
                f"{caso['pergunta']} -> {saida['documento']}, esperado {caso['documento_esperado']}"
            )
    assert len(perdidas) <= TETO_MUDEZ_BEM_FORMADA, "\n".join(perdidas)


def test_nenhum_ataque_chega_ao_modelo_de_linguagem(motor, espiao, corpus):
    """O que o enunciado cobra e que o modelo nao seja chamado sem documento.

    Vale para parafrase e seletor, os dois mecanismos fechados. A co-citacao
    chega ao modelo por desenho - ela tem documento aprovado, o do defeito que a
    pergunta nomeou.
    """
    casos = [c for c in corpus["ataques"] if c["mecanismo"] in {"parafrase", "seletor"}]
    antes = len(espiao.chamadas)
    for caso in casos:
        motor.perguntar(caso["pergunta"], documento=caso.get("documento_seletor") or None)
    assert len(espiao.chamadas) == antes, (
        "o modelo foi chamado para um defeito sem procedimento cadastrado"
    )
