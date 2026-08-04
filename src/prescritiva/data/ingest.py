"""Ingestao do historico de eventos: CSV bruto -> tabela analitica + SQLite."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from prescritiva.config import load_settings
from prescritiva.data.schema import UNIT_DUPLICATES

# A vibracao escala com a rotacao, entao comparar um evento de 500 rpm com o
# historico de 2000 rpm nao diz nada: o regime particiona a busca.
# O ensaio operou em quatro rotacoes fixas, verificado em scripts/eda.py, o que
# permite pareamento exato em vez de tolerancia percentual. Uma rotacao fora da
# lista cai no valor conhecido mais proximo, para o sistema nao quebrar em campo.
RPM_CONHECIDOS: tuple[float, ...] = (0.0, 500.0, 1000.0, 2000.0)


def normalize_fault(fault: str, pattern: str) -> str:
    """Remove o sufixo de campanha de coleta ("cocked_rotor_2" -> "cocked_rotor").

    Cuidado deliberado com "desbalanceado_1parafuso": o "1parafuso" faz parte do
    nome do defeito e nao termina em "_<digitos>", entao a regex nao o toca.
    """
    return re.sub(pattern, "", fault)


def rpm_regime(rpm: float) -> str:
    mais_proximo = min(RPM_CONHECIDOS, key=lambda alvo: abs(alvo - rpm))
    return f"{int(mais_proximo)}rpm"


def load_raw(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["created_at"])


def prepare(df: pd.DataFrame, *, suffix_pattern: str, operational_states: list[str]) -> pd.DataFrame:
    out = df.drop(columns=[c for c in UNIT_DUPLICATES if c in df.columns]).copy()
    out["fault_original"] = out["fault"]
    out["fault"] = out["fault"].map(lambda f: normalize_fault(f, suffix_pattern))
    # A campanha e o grupo de vazamento: dentro dela os eventos sao coletados a
    # segundos de distancia e sao quase identicos. Ela existe aqui para que a
    # avaliacao possa separar treino e teste por campanha, nunca ao acaso.
    out["campanha"] = (
        out["fault_original"].str.extract(suffix_pattern, expand=False).fillna("1").astype(int)
    )
    out["regime_rpm"] = out["rpm"].map(rpm_regime)
    out["e_defeito"] = ~out["fault"].isin(operational_states)
    return out


def write_sqlite(df: pd.DataFrame, database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as conn:
        df.to_sql("eventos", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_eventos_fault ON eventos(fault)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_eventos_regime ON eventos(regime_rpm)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_eventos_created ON eventos(created_at)")


def run() -> pd.DataFrame:
    settings = load_settings()
    settings.paths.ensure()
    df = load_raw(settings.paths.raw_csv)
    prepared = prepare(
        df,
        suffix_pattern=settings.ingest["campaign_suffix_pattern"],
        operational_states=settings.ingest["operational_states"],
    )
    prepared.to_parquet(settings.paths.processed_dir / "eventos.parquet", index=False)
    write_sqlite(prepared, settings.paths.database)
    return prepared
