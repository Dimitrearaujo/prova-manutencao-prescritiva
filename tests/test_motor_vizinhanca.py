"""Vizinhanca temporal e consenso: o que o indice pode e nao pode contar como voto.

Dois defeitos que se somavam. O primeiro e vazamento: consultar o indice de
producao com um evento tirado do proprio historico excluia so o id, e a leitura
gravada dois segundos depois continuava la, a distancia ~0. O segundo e o peso
1/(d + 1e-6): esse vizinho unico levava 0,99998 do peso e o kNN virava 1-NN,
com o portao de consenso aprovando vizinhancas que a maioria simples reprovava.

Juntos faziam a demonstracao rodar contra um sistema diferente do que foi
medido em scripts/evaluate.py - exatamente o vazamento que o projeto condena.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prescritiva.config import load_settings
from prescritiva.similarity.index import IndiceSimilaridade, _pesos_por_distancia


@pytest.fixture(scope="module")
def indice() -> IndiceSimilaridade:
    settings = load_settings()
    if not (settings.paths.index_dir / "indice_similaridade.joblib").exists():
        pytest.skip("execute scripts/build_index.py antes")
    return IndiceSimilaridade.carregar(settings.paths.index_dir)


@pytest.fixture(scope="module")
def eventos() -> pd.DataFrame:
    caminho = load_settings().paths.processed_dir / "eventos.parquet"
    if not caminho.exists():
        pytest.skip("execute scripts/ingest.py antes")
    return pd.read_parquet(caminho)


def test_peso_de_um_vizinho_nao_engole_a_vizinhanca():
    """Um vizinho a distancia zero entre 24 a distancia tipica.

    Com 1/(d + 1e-6) ele levava mais de 99,9% do peso. O piso relativo limita a
    fatia dele, e os outros 24 continuam existindo na conta.
    """
    distancias = [0.0] + [1.0] * 24
    pesos = _pesos_por_distancia(distancias)

    assert pytest.approx(sum(pesos), abs=1e-9) == 1.0
    assert pesos[0] < 0.5
    assert min(pesos[1:]) > 0.01


def test_peso_nao_estoura_com_vizinhanca_inteira_a_distancia_zero():
    pesos = _pesos_por_distancia([0.0] * 5)

    assert all(np.isfinite(p) for p in pesos)
    assert pytest.approx(sum(pesos), abs=1e-9) == 1.0


def test_janela_temporal_tira_o_gemeo_do_evento_consultado(indice, eventos):
    """O caso que a demo do painel exercita: evento sorteado do historico."""
    bloco = eventos[eventos["fault"] == "desalinhado"].sort_values("created_at")
    linha = bloco.iloc[len(bloco) // 2]
    quadro = linha.to_frame().T

    so_o_id = indice.consultar(quadro, excluir_ids={int(linha["id"])})
    com_janela = indice.consultar(quadro, momento=linha["created_at"], janela_s=300.0)

    assert so_o_id.vizinhos and com_janela.vizinhos
    # O gemeo esta a segundos do evento e a distancia praticamente zero.
    assert min(v.distancia for v in so_o_id.vizinhos) < 0.01
    assert min(v.distancia for v in com_janela.vizinhos) > 0.01

    momento = pd.Timestamp(linha["created_at"])
    for vizinho in com_janela.vizinhos:
        distancia_no_tempo = abs((pd.Timestamp(vizinho.created_at) - momento).total_seconds())
        assert distancia_no_tempo > 300.0

    # A vizinhanca continua completa: quem sai e substituido por quem esta fora
    # da janela, e nao por um consenso apurado sobre meia duzia de sobreviventes.
    assert len(com_janela.vizinhos) == len(so_o_id.vizinhos)


def test_confianca_ponderada_deixa_de_ser_certeza_falsa(indice, eventos):
    """Sem a janela, a mediana da confianca era 0,99998 em consulta ao historico."""
    amostra = eventos.sample(n=40, random_state=11)
    confiancas = [
        indice.consultar(
            linha.to_frame().T, momento=linha["created_at"], janela_s=300.0
        ).confianca
        for _, linha in amostra.iterrows()
    ]

    assert float(np.median(confiancas)) < 0.95


def test_consenso_simples_acompanha_a_contagem_de_votos(indice, eventos):
    linha = eventos.sample(1, random_state=3).iloc[0]
    resultado = indice.consultar(
        linha.to_frame().T, momento=linha["created_at"], janela_s=300.0
    )

    assert sum(resultado.votos.values()) == len(resultado.vizinhos)
    esperado = resultado.votos[resultado.fault_predominante] / len(resultado.vizinhos)
    assert resultado.consenso_simples == pytest.approx(esperado)


def test_portao_de_consenso_roda_sobre_o_voto_simples(indice, eventos):
    """Nenhum evento aceito pode ter maioria simples abaixo do minimo.

    E o caso que o portao antigo deixava passar: voto bruto de 44% aprovado
    porque o peso do vizinho mais proximo somava 99,999%.
    """
    amostra = eventos.sample(n=60, random_state=17)
    aceitos_indevidos = []
    for _, linha in amostra.iterrows():
        resultado = indice.consultar(
            linha.to_frame().T, momento=linha["created_at"], janela_s=300.0
        )
        if resultado.reconhecido and resultado.consenso_simples < indice.min_consensus:
            aceitos_indevidos.append((int(linha["id"]), resultado.consenso_simples))

    assert not aceitos_indevidos
