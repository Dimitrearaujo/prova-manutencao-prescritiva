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

ALL_COLUMNS: tuple[str, ...] = SENSOR_COLUMNS + DERIVED_COLUMNS

# Colunas calculadas mas mantidas fora do espaco de busca. A analise em
# scripts/eda.py mostrou que nenhuma delas mede o defeito, e scripts/ablacao_features.py
# confirmou que remove-las nao piora nada. Continuam sendo calculadas porque a
# ablacao precisa poder reintroduzi-las para reproduzir a comparacao.
EXCLUIDAS: tuple[str, ...] = (
    # 14 valores distintos, 61 Hz em 61% das linhas, inclusive com o motor
    # parado. E a frequencia da rede eletrica, nao uma medicao de vibracao. E o
    # IQR que a escala aqui varia de 1.0 a 44.0 entre regimes: onde ele e
    # pequeno, a coluna sozinha vale dezenas de unidades de distancia.
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    # Derivadas da coluna acima: se a origem e artefato, a razao tambem e.
    "ordem_z",
    "ordem_x",
    # Correlaciona ate 0.93 com a posicao no bloco de gravacao, subindo em uns e
    # caindo em outros, e os blocos mais correlacionados sao todos da segunda
    # rodada: e aquecimento ambiente da sessao. E separa os defeitos pior do que
    # varia dentro de cada um (razao entre/dentro 0.72).
    "temperature_c",
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(c for c in ALL_COLUMNS if c not in EXCLUIDAS)


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

    selecionadas = list(colunas if colunas is not None else FEATURE_COLUMNS)
    return out[selecionadas].replace([np.inf, -np.inf], np.nan).fillna(0.0)
