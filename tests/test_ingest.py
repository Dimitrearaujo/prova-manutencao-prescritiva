import pandas as pd
import pytest

from prescritiva.config import load_settings
from prescritiva.data.ingest import normalize_fault, prepare, rpm_regime
from prescritiva.data.schema import UNIT_DUPLICATES

PADRAO = load_settings().ingest["campaign_suffix_pattern"]


@pytest.mark.parametrize(
    ("bruto", "esperado"),
    [
        ("cocked_rotor_2", "cocked_rotor"),
        ("rolamento_inner_2", "rolamento_inner"),
        ("normal_2", "normal"),
        ("desbalanceado_1parafuso_3", "desbalanceado_1parafuso"),
        ("desalinhado", "desalinhado"),
    ],
)
def test_sufixo_de_campanha_some(bruto, esperado):
    assert normalize_fault(bruto, PADRAO) == esperado


def test_numero_dentro_do_nome_do_defeito_e_preservado():
    # "1parafuso" e a severidade do desbalanceamento, nao campanha de coleta.
    assert normalize_fault("desbalanceado_1parafuso", PADRAO) == "desbalanceado_1parafuso"


@pytest.mark.parametrize(
    ("rpm", "esperado"),
    [(0.0, "0rpm"), (500.0, "500rpm"), (1000.0, "1000rpm"), (2000.0, "2000rpm")],
)
def test_regime_das_rotacoes_do_ensaio(rpm, esperado):
    assert rpm_regime(rpm) == esperado


def test_rotacao_desconhecida_cai_no_regime_mais_proximo():
    assert rpm_regime(1950.0) == "2000rpm"
    assert rpm_regime(520.0) == "500rpm"


def _amostra() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "created_at": "2026-06-01",
                "fault": "cocked_rotor_2",
                "rpm": 2000.0,
                "z_rms_velocity_in_s": 0.06,
                "z_rms_velocity_mm_s": 1.52,
                "temperature_f": 76.4,
                "temperature_c": 24.7,
            },
            {
                "id": 2,
                "created_at": "2026-06-01",
                "fault": "normal",
                "rpm": 0.0,
                "z_rms_velocity_in_s": 0.04,
                "z_rms_velocity_mm_s": 1.08,
                "temperature_f": 74.0,
                "temperature_c": 23.3,
            },
        ]
    )


def test_prepare_descarta_colunas_de_unidade_duplicada():
    resultado = prepare(_amostra(), suffix_pattern=PADRAO, operational_states=["normal"])
    for coluna in UNIT_DUPLICATES:
        assert coluna not in resultado.columns
    assert "z_rms_velocity_mm_s" in resultado.columns
    assert "temperature_c" in resultado.columns


def test_prepare_marca_campanha_e_preserva_rotulo_original():
    resultado = prepare(_amostra(), suffix_pattern=PADRAO, operational_states=["normal"])
    assert list(resultado["campanha"]) == [2, 1]
    assert list(resultado["fault_original"]) == ["cocked_rotor_2", "normal"]
    assert list(resultado["fault"]) == ["cocked_rotor", "normal"]


def test_estado_operacional_nao_e_defeito():
    resultado = prepare(_amostra(), suffix_pattern=PADRAO, operational_states=["normal"])
    assert list(resultado["e_defeito"]) == [True, False]
