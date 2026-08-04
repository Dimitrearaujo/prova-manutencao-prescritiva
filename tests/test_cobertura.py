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


def test_mencao_solta_nao_conta_como_cobertura(base):
    """O procedimento de rolamentos lista "Ventiladores" entre os equipamentos.

    Se a cobertura fosse decidida por presenca do termo no corpo, uma falha de
    ventoinha receberia o procedimento de rolamento e o tecnico seria mandado
    trocar um mancal por causa de uma palavra numa lista.
    """
    doc1 = next(d for d in base.documentos if d.nome == "Doc1")
    assert "entilador" in doc1.escopo
    cobertura = base.cobertura("ventoinha", "ventoinha pas helice", "Falha em ventoinha")
    assert not cobertura.coberto
