"""Rotacao fora da grade calibrada nao recebe diagnostico.

O ensaio operou em 0, 500, 1000 e 2000 rpm. Encaixar qualquer rotacao no valor
mais proximo, sem limite, fazia um evento de 12000 rpm ser comparado com o
historico de 2000 rpm e sair com defeito identificado e 100% de confianca - e
nada na resposta denunciava o encaixe, porque o dataclass nao carregava o rpm.
"""

from __future__ import annotations

import pytest

from prescritiva.config import load_settings
from prescritiva.data.ingest import REGIME_FORA_DA_GRADE, regime_aproximado, rpm_regime
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.knowledge.store import BaseConhecimento
from prescritiva.llm.deterministico import GeradorDeterministico
from prescritiva.similarity.index import IndiceSimilaridade


@pytest.fixture(scope="module")
def motor() -> MotorDiagnostico:
    settings = load_settings()
    if not (settings.paths.index_dir / "indice_similaridade.joblib").exists():
        pytest.skip("execute scripts/build_index.py e build_knowledge.py antes")
    return MotorDiagnostico(
        indice=IndiceSimilaridade.carregar(settings.paths.index_dir),
        base=BaseConhecimento.carregar(settings.paths.index_dir),
        database=settings.paths.database,
        gerador=GeradorDeterministico(),
        top_k=settings.knowledge["retrieval_top_k"],
    )


# Evento real de rolamento_inner a 2000 rpm, do proprio historico. E o mesmo
# vetor usado com rotacoes diferentes, para que a rotacao seja a unica variavel.
EVENTO = {
    "rpm": 2000.0,
    "z_rms_velocity_mm_s": 1.83,
    "x_rms_velocity_mm_s": 1.24,
    "temperature_c": 27.6,
    "z_peak_acceleration_g": 1.62,
    "x_peak_acceleration_g": 1.11,
    "z_peak_vel_comp_freq_hz": 61.0,
    "x_peak_vel_comp_freq_hz": 61.0,
    "z_rms_acceleration_g": 0.32,
    "x_rms_acceleration_g": 0.24,
    "z_kurtosis": 3.1,
    "x_kurtosis": 3.4,
    "z_crest_factor": 4.6,
    "x_crest_factor": 4.4,
    "z_peak_velocity_mm_s": 3.9,
    "x_peak_velocity_mm_s": 2.7,
    "z_high_freq_rms_accel_g": 0.12,
    "x_high_freq_rms_accel_g": 0.09,
}


@pytest.mark.parametrize("rpm", [12000.0, 1500.0, 4000.0, -50.0, 300.0])
def test_rotacao_fora_da_tolerancia_nao_tem_regime(rpm):
    assert rpm_regime(rpm) == REGIME_FORA_DA_GRADE


@pytest.mark.parametrize(
    ("rpm", "esperado"),
    [(0.0, "0rpm"), (500.0, "500rpm"), (1000.0, "1000rpm"), (2000.0, "2000rpm"),
     (1950.0, "2000rpm"), (520.0, "500rpm"), (15.0, "0rpm")],
)
def test_rotacao_dentro_da_tolerancia_mantem_o_regime(rpm, esperado):
    assert rpm_regime(rpm) == esperado


def test_sinalizador_de_regime_aproximado():
    assert regime_aproximado(1950.0) is True
    assert regime_aproximado(2000.0) is False
    # Fora da grade nao e "aproximado", e sem regime nenhum.
    assert regime_aproximado(12000.0) is False


def test_evento_com_rotacao_absurda_nao_recebe_diagnostico(motor):
    diagnostico = motor.diagnosticar({**EVENTO, "rpm": 12000.0})

    assert diagnostico.situacao == "padrao_desconhecido"
    assert diagnostico.fault is None
    assert not diagnostico.instrucoes
    assert diagnostico.confianca == 0.0
    assert "12000 rpm" in diagnostico.mensagem
    assert "nao tem historico comparavel" in diagnostico.mensagem


def test_diagnostico_carrega_a_rotacao_bruta_e_o_aviso_de_encaixe(motor):
    """O aviso precisa viajar pela API e pela fila, nao so pela tela."""
    encaixado = motor.diagnosticar({**EVENTO, "rpm": 1900.0})

    assert encaixado.rpm == 1900.0
    assert encaixado.regime_aproximado is True
    assert encaixado.regime_rpm == "2000rpm"
    assert "1900 rpm" in encaixado.mensagem

    exato = motor.diagnosticar(EVENTO)
    assert exato.rpm == 2000.0
    assert exato.regime_aproximado is False
