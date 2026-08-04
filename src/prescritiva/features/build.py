"""Construcao da matriz de features.

Esta funcao roda em dois lugares: ao indexar as 144 mil linhas do historico e
ao diagnosticar um unico evento novo. E a mesma funcao de proposito, para que
nao exista divergencia entre o que foi indexado e o que e consultado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prescritiva.data.schema import SENSOR_COLUMNS

_EPS = 1e-6

# Features derivadas de analise de vibracao. As razoes X/Z separam defeito
# radial de axial, e a "ordem" (frequencia de pico dividida pela frequencia de
# rotacao) e o indicador classico: 1x aponta desbalanceamento, 2x aponta
# desalinhamento, ordens altas apontam rolamento.
DERIVED_COLUMNS: tuple[str, ...] = (
    "razao_vel_xz",
    "razao_accel_xz",
    "razao_alta_freq_z",
    "razao_alta_freq_x",
    "ordem_z",
    "ordem_x",
    "fator_crista_medio",
    "kurtosis_media",
)

FEATURE_COLUMNS: tuple[str, ...] = SENSOR_COLUMNS + DERIVED_COLUMNS


def build_features(df: pd.DataFrame, colunas: tuple[str, ...] | None = None) -> pd.DataFrame:
    freq_rotacao = df["rpm"] / 60.0

    out = df[list(SENSOR_COLUMNS)].copy()
    out["razao_vel_xz"] = df["x_rms_velocity_mm_s"] / (df["z_rms_velocity_mm_s"] + _EPS)
    out["razao_accel_xz"] = df["x_rms_acceleration_g"] / (df["z_rms_acceleration_g"] + _EPS)
    out["razao_alta_freq_z"] = df["z_high_freq_rms_accel_g"] / (df["z_rms_acceleration_g"] + _EPS)
    out["razao_alta_freq_x"] = df["x_high_freq_rms_accel_g"] / (df["x_rms_acceleration_g"] + _EPS)
    out["ordem_z"] = df["z_peak_vel_comp_freq_hz"] / (freq_rotacao + _EPS)
    out["ordem_x"] = df["x_peak_vel_comp_freq_hz"] / (freq_rotacao + _EPS)
    out["fator_crista_medio"] = (df["z_crest_factor"] + df["x_crest_factor"]) / 2.0
    out["kurtosis_media"] = (df["z_kurtosis"] + df["x_kurtosis"]) / 2.0

    # Motor parado tem rpm zero: a "ordem" nao existe nesse regime e viraria
    # um valor gigante que domina a distancia. Zerar e o comportamento correto.
    parado = df["rpm"] <= 0
    out.loc[parado, ["ordem_z", "ordem_x"]] = 0.0

    selecionadas = list(colunas or FEATURE_COLUMNS)
    return out[selecionadas].replace([np.inf, -np.inf], np.nan).fillna(0.0)
