"""Testes da leitura de contexto operacional.

A funcao e pura de proposito: as quatro leituras possiveis sao verificadas com
distribuicoes montadas a mao, sem depender do indice nem do banco.
"""

from __future__ import annotations

from prescritiva.diagnosis.contexto import ATIPICO, INEDITO, SEM_HISTORICO, TIPICO, contexto_regime


def test_regime_dominante_e_lido_como_esperado():
    contexto = contexto_regime({"500rpm": 2186, "1000rpm": 1000, "2000rpm": 931}, "500rpm", "Desalinhamento")
    assert contexto.alinhamento == TIPICO
    assert contexto.regime_dominante == "500rpm"
    assert "500 rpm" in contexto.leitura


def test_regime_minoritario_vira_alerta_com_o_pico_nomeado():
    """O caso que o enunciado chama de insight: mesmo defeito, rotacao fora do padrao."""
    contexto = contexto_regime({"500rpm": 2186, "1000rpm": 1000, "2000rpm": 931}, "2000rpm", "Desalinhamento")
    assert contexto.alinhamento == ATIPICO
    assert contexto.regime_dominante == "500rpm"
    assert "2000 rpm" in contexto.leitura and "500 rpm" in contexto.leitura


def test_distribuicao_equilibrada_nao_alarma():
    """Com os regimes proximos entre si nao ha o que sinalizar.

    Vale como guarda contra alerta barato: no historico do ensaio quase todo
    defeito foi gravado nas tres rotacoes em proporcao parecida, e um limiar
    absoluto acusaria fora do padrao em quase todo diagnostico.
    """
    contexto = contexto_regime({"500rpm": 4000, "1000rpm": 4000, "2000rpm": 4000}, "2000rpm", "Polia")
    assert contexto.alinhamento == TIPICO


def test_regime_sem_nenhum_registro_do_padrao():
    contexto = contexto_regime({"500rpm": 4000, "1000rpm": 4000}, "2000rpm", "Correia")
    assert contexto.alinhamento == INEDITO
    assert contexto.ocorrencias_no_regime == 0
    assert "nunca foi registrado" in contexto.leitura


def test_padrao_sem_historico_nao_inventa_leitura():
    contexto = contexto_regime({}, "1000rpm", "Defeito novo")
    assert contexto.alinhamento == SEM_HISTORICO
    assert contexto.total == 0
    assert contexto.regime_dominante is None


def test_fracoes_somam_o_que_a_tela_desenha():
    por_regime = {"500rpm": 3000, "1000rpm": 1000}
    contexto = contexto_regime(por_regime, "500rpm", "Rolamento")
    assert contexto.total == 4000
    assert contexto.fracao_no_regime == 0.75
    assert contexto.fracao_dominante == 0.75
