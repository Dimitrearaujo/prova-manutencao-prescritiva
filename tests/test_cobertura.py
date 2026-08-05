"""Fixa o mapa de cobertura documental.

A associacao defeito -> documento nao esta escrita em lugar nenhum do codigo:
ela e descoberta em tempo de execucao pela regra do termo-chave. Este teste
existe para travar essa descoberta contra a leitura humana dos seis PDFs, e para
quebrar se alguem mexer no limiar, no radical ou no catalogo e o mapa mudar.

Os dois defeitos sem documento nao sao falha do sistema: sao o caso que o
enunciado manda tratar informando que o problema ainda nao esta documentado.
"""

from __future__ import annotations

import pytest

from prescritiva.config import load_catalog, load_settings
from prescritiva.knowledge.store import BaseConhecimento
from prescritiva.text import normalize

ESPERADO: dict[str, str | None] = {
    "rolamento_inner": "Doc1",
    "rolamento_outer": "Doc1",
    "rolamento_ball": "Doc1",
    "rolamento_combination": "Doc1",
    "desalinhado": "Doc2",
    "desbalanceado_1parafuso": "Doc3",
    "correia": "Doc4",
    "polia": "Doc5",
    "cocked_rotor": "Doc6",
    "eccentric_rotor": None,
    "ventoinha": None,
}


@pytest.fixture(scope="module")
def base() -> BaseConhecimento:
    indice = load_settings().paths.index_dir
    if not (indice / "base_conhecimento.json").exists():
        pytest.skip("execute scripts/build_knowledge.py antes")
    return BaseConhecimento.carregar(indice)


@pytest.fixture(scope="module")
def catalogo() -> dict:
    return load_catalog()["faults"]


@pytest.mark.parametrize(("fault", "documento"), ESPERADO.items())
def test_mapa_de_cobertura(base, catalogo, fault, documento):
    entrada = catalogo[fault]
    cobertura = base.cobertura(entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"])
    assert cobertura.documento == documento
    assert cobertura.coberto is (documento is not None)


def test_limiar_fica_numa_faixa_vazia(base, catalogo):
    """O limiar precisa separar por folga, nao no fio da navalha.

    Se a menor aderencia coberta chegar perto da maior aderencia rejeitada, uma
    troca de documento faria o mapa virar sozinho.
    """
    cobertos, rejeitados = [], []
    for fault, esperado in ESPERADO.items():
        entrada = catalogo[fault]
        ranking = base.pontuar_cobertura(entrada["termo_chave"], entrada["termos_busca"])
        melhor = ranking[0][0] if ranking else 0.0
        (cobertos if esperado else rejeitados).append(melhor)
        # O segundo colocado tambem precisa ficar longe do limiar.
        if len(ranking) > 1:
            rejeitados.append(ranking[1][0])

    limiar = load_settings().knowledge["coverage_min_score"]
    assert min(cobertos) > limiar * 1.5
    assert max(rejeitados) < limiar * 0.8


def test_o_acervo_real_nao_discrimina_escopo_de_corpo(base):
    """Registra que a regra do escopo NAO corrige nenhum erro neste acervo.

    Os seis procedimentos sao monotematicos e nomeiam o componente no titulo, e
    nenhum cita no corpo um componente tratado por outro. O unico termo cruzado e
    "ventilador", no procedimento de rolamentos - e ele nao casa com "ventoinha"
    porque os radicais diferem (tests/test_text.py verifica).

    Este teste existe porque a documentacao do projeto ja afirmou o contrario, e
    a afirmacao caia em cinco linhas de codigo. Se um documento novo criar o caso
    discriminante, este teste falha e obriga a rever a redacao em vez de deixar
    uma justificativa envelhecer sozinha.
    """
    corpo_doc1 = normalize(" ".join(t.texto for t in base.trechos if t.documento == "Doc1"))
    for termo in ("polia", "correia", "acoplamento", "desalinha", "excentric"):
        assert termo not in corpo_doc1, (
            f'"{termo}" passou a aparecer no corpo do Doc1: o acervo agora tem caso '
            "discriminante e a redacao da secao 3.6 da ARQUITETURA precisa ser revista"
        )


def test_mencao_no_corpo_nao_concede_cobertura(base):
    """A garantia da regra, provada num caso CONSTRUIDO.

    Como o acervo entregue nao produz o conflito, o caso e montado aqui: um
    procedimento que TRATA de desalinhamento e apenas MENCIONA ventoinha varias
    vezes no corpo. A cobertura de ventoinha nao pode cair nele.
    """
    from prescritiva.knowledge.extract import ExtractedDocument

    documento = ExtractedDocument(
        nome="DocSintetico",
        arquivo="DocSintetico.pdf",
        paginas=1,
        metodo="sintetico-de-teste",
        texto=(
            "Procedimento para Correcao de Desalinhamento em Motor Eletrico\n"
            "1. Objetivo\n"
            "Corrigir desalinhamento entre o eixo do motor e a carga acionada.\n"
            "2. Sintomas\n"
            "O desalinhamento aparece junto com vibracao da ventoinha, ruido na "
            "ventoinha e desgaste da ventoinha. Uma ventoinha trincada agrava o "
            "quadro. Inspecione a ventoinha antes de alinhar.\n"
        ),
    )
    sintetica = BaseConhecimento(score_minimo_cobertura=base.score_minimo_cobertura).indexar(
        [documento]
    )

    corpo = normalize(" ".join(t.texto for t in sintetica.trechos))
    assert corpo.count("ventoinha") >= 4, "o caso construido precisa citar ventoinha no corpo"

    cobertura = sintetica.cobertura("ventoinha", "ventoinha pas helice", "Falha em ventoinha")
    assert not cobertura.coberto, "mencao no corpo nao pode conceder cobertura"

    assert sintetica.cobertura(
        "desalinhamento", "desalinhamento eixo acoplamento", "Desalinhamento de eixo"
    ).coberto, "o documento continua cobrindo o defeito que ele de fato trata"
