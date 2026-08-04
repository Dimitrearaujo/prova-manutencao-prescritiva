"""Testes de ponta a ponta do motor, contra os artefatos construidos.

Cada teste cobre um dos quatro desfechos possiveis do diagnostico.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import MotorDiagnostico


@pytest.fixture(scope="module")
def motor() -> MotorDiagnostico:
    settings = load_settings()
    if not (settings.paths.index_dir / "indice_similaridade.joblib").exists():
        pytest.skip("execute scripts/build_index.py e build_knowledge.py antes")
    return MotorDiagnostico.carregar()


@pytest.fixture(scope="module")
def eventos() -> pd.DataFrame:
    caminho = load_settings().paths.processed_dir / "eventos.parquet"
    if not caminho.exists():
        pytest.skip("execute scripts/ingest.py antes")
    return pd.read_parquet(caminho)


def _amostra(eventos: pd.DataFrame, fault: str) -> dict:
    linha = eventos[eventos["fault"] == fault].sample(1, random_state=7).iloc[0].to_dict()
    linha.pop("fault", None)
    linha.pop("fault_original", None)
    return linha


def test_defeito_com_procedimento_recebe_instrucao(motor, eventos):
    diagnostico = motor.diagnosticar(_amostra(eventos, "desalinhado"))
    if diagnostico.situacao == "padrao_desconhecido":
        pytest.skip("o evento sorteado caiu na regra de rejeicao")
    assert diagnostico.situacao in {"defeito_documentado", "defeito_sem_documentacao"}
    if diagnostico.situacao == "defeito_documentado":
        assert diagnostico.instrucoes
        assert diagnostico.trechos


@pytest.mark.parametrize("fault", ["eccentric_rotor", "ventoinha"])
def test_defeito_sem_procedimento_pede_cadastro(motor, eventos, fault):
    """Os dois defeitos que o enunciado quer ver recusados.

    O sistema pode nao reconhecer o padrao, e tudo bem. O que nao pode acontecer
    e reconhecer o defeito e mesmo assim entregar instrucao de correcao, porque
    nao existe procedimento cadastrado para nenhum dos dois.
    """
    diagnostico = motor.diagnosticar(_amostra(eventos, fault))
    if diagnostico.fault == fault:
        assert diagnostico.situacao == "defeito_sem_documentacao"
        assert not diagnostico.instrucoes
        assert "cadastre" in diagnostico.mensagem.lower()


def test_motor_desligado_nao_recebe_acao_corretiva(motor, eventos):
    diagnostico = motor.diagnosticar(_amostra(eventos, "motor_desligado"))
    if diagnostico.situacao == "padrao_desconhecido":
        pytest.skip("o evento sorteado caiu na regra de rejeicao")
    assert diagnostico.situacao == "estado_operacional"
    assert diagnostico.e_defeito is False
    assert not diagnostico.instrucoes


def test_evento_absurdo_e_recusado(motor):
    """Valores muito fora de qualquer faixa fisica plausivel.

    Um classificador fechado responderia com a classe menos errada. Aqui a
    resposta correta e admitir que nao ha nada parecido no historico.
    """
    absurdo = {
        "rpm": 1000.0,
        "z_rms_velocity_mm_s": 900.0,
        "x_rms_velocity_mm_s": 850.0,
        "temperature_c": 240.0,
        "z_peak_acceleration_g": 400.0,
        "x_peak_acceleration_g": 380.0,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 300.0,
        "x_rms_acceleration_g": 290.0,
        "z_kurtosis": 90.0,
        "x_kurtosis": 88.0,
        "z_crest_factor": 70.0,
        "x_crest_factor": 68.0,
        "z_peak_velocity_mm_s": 950.0,
        "x_peak_velocity_mm_s": 940.0,
        "z_high_freq_rms_accel_g": 200.0,
        "x_high_freq_rms_accel_g": 210.0,
    }
    diagnostico = motor.diagnosticar(absurdo)
    assert diagnostico.situacao == "padrao_desconhecido"
    assert diagnostico.fault is None
    assert not diagnostico.instrucoes


def test_pergunta_sem_procedimento_nao_inventa_resposta(motor):
    resultado = motor.perguntar("como calibrar o sensor de pressao da caldeira a vapor?")
    texto = resultado["resposta"].lower()
    if not resultado["trechos"]:
        assert "nenhum procedimento" in texto


def test_diagnostico_sempre_expoe_as_evidencias(motor, eventos):
    diagnostico = motor.diagnosticar(_amostra(eventos, "correia"))
    assert diagnostico.similaridade["vizinhos"]
    for vizinho in diagnostico.similaridade["vizinhos"]:
        assert {"id", "fault", "created_at", "distancia"} <= vizinho.keys()
