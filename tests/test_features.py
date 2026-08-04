import pandas as pd
import pytest

from prescritiva.features.build import ALL_COLUMNS, EXCLUIDAS, FEATURE_COLUMNS, build_features


def _evento(**sobrescritas) -> pd.DataFrame:
    base = {
        "rpm": 1000.0,
        "z_rms_velocity_mm_s": 1.517,
        "x_rms_velocity_mm_s": 2.0,
        "temperature_c": 24.69,
        "z_peak_acceleration_g": 0.484,
        "x_peak_acceleration_g": 0.631,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 0.09,
        "x_rms_acceleration_g": 0.114,
        "z_kurtosis": 2.392,
        "x_kurtosis": 2.77,
        "z_crest_factor": 3.747,
        "x_crest_factor": 4.269,
        "z_peak_velocity_mm_s": 2.146,
        "x_peak_velocity_mm_s": 2.829,
        "z_high_freq_rms_accel_g": 0.129,
        "x_high_freq_rms_accel_g": 0.147,
    }
    return pd.DataFrame([{**base, **sobrescritas}])


def test_colunas_de_saida_sao_estaveis():
    assert tuple(build_features(_evento()).columns) == FEATURE_COLUMNS


def test_razao_entre_eixos():
    features = build_features(_evento(x_rms_velocity_mm_s=3.0, z_rms_velocity_mm_s=1.5))
    assert features["razao_vel_xz"].iloc[0] == pytest.approx(2.0, rel=1e-4)


def test_ordem_e_a_frequencia_dividida_pela_rotacao():
    # 61 Hz a 1000 rpm: a rotacao e 16.67 Hz, entao a ordem e 3.66.
    features = build_features(_evento(rpm=1000.0, z_peak_vel_comp_freq_hz=61.0), ALL_COLUMNS)
    assert features["ordem_z"].iloc[0] == pytest.approx(3.66, abs=0.01)


def test_motor_parado_nao_gera_ordem_infinita():
    """Sem esta guarda, dividir por rotacao zero produziria um valor enorme
    que dominaria a distancia e tornaria os 347 registros de motor desligado
    vizinhos de nada."""
    features = build_features(_evento(rpm=0.0), ALL_COLUMNS)
    assert features["ordem_z"].iloc[0] == 0.0
    assert features["ordem_x"].iloc[0] == 0.0


def test_confounds_ficam_fora_do_espaco_de_busca():
    """Frequencia de pico e temperatura sao calculadas mas nao entram na busca.

    A frequencia marca 61 Hz ate com o motor parado, o que e a rede eletrica; a
    temperatura acompanha a posicao no bloco de gravacao, o que e aquecimento
    ambiente. Nenhuma das duas mede o defeito.
    """
    for coluna in EXCLUIDAS:
        assert coluna not in FEATURE_COLUMNS
        assert coluna in ALL_COLUMNS


def test_nenhum_valor_invalido_sobrevive():
    features = build_features(_evento(rpm=0.0, z_rms_velocity_mm_s=0.0, z_rms_acceleration_g=0.0))
    assert features.notna().all().all()
    assert (features.abs() < 1e9).all().all()


def test_selecao_de_subconjunto_de_colunas():
    escolhidas = ("z_kurtosis", "razao_vel_xz")
    assert tuple(build_features(_evento(), escolhidas).columns) == escolhidas


def test_mesma_entrada_produz_a_mesma_saida():
    pd.testing.assert_frame_equal(build_features(_evento()), build_features(_evento()))
