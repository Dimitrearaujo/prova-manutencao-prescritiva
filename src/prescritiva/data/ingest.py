"""Ingestao do historico de eventos: CSV bruto -> tabela analitica + SQLite."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from prescritiva.config import load_settings
from prescritiva.data.schema import UNIT_DUPLICATES

# A vibracao escala com a rotacao, entao comparar um evento de 500 rpm com o
# historico de 2000 rpm nao diz nada: o regime particiona a busca.
# O ensaio operou em quatro rotacoes fixas, verificado em scripts/eda.py.
RPM_CONHECIDOS: tuple[float, ...] = (0.0, 500.0, 1000.0, 2000.0)

# Tolerancia em torno de cada regime. Fora dela nao existe historico comparavel
# e o evento nao pode receber diagnostico: encaixar no vizinho mais proximo sem
# limite fazia 12000 rpm ser comparado com o bloco de 2000 rpm e sair com 100%
# de confianca. A parte relativa acompanha a escala do regime; o piso absoluto
# existe porque 10% de 0 rpm seria tolerancia zero, e tambem mantem rotacao
# negativa fora de qualquer regime.
RPM_TOLERANCIA_RELATIVA = 0.10
RPM_TOLERANCIA_ABSOLUTA = 25.0
REGIME_FORA_DA_GRADE = "fora_da_grade"


def _tolerancia_regime(alvo: float) -> float:
    return max(alvo * RPM_TOLERANCIA_RELATIVA, RPM_TOLERANCIA_ABSOLUTA)


def normalize_fault(fault: str, pattern: str) -> str:
    """Remove o sufixo de campanha de coleta ("cocked_rotor_2" -> "cocked_rotor").

    Cuidado deliberado com "desbalanceado_1parafuso": o "1parafuso" faz parte do
    nome do defeito e nao termina em "_<digitos>", entao a regex nao o toca.
    """
    return re.sub(pattern, "", fault)


def rpm_regime(rpm: float) -> str:
    """Regime de comparacao do evento, ou REGIME_FORA_DA_GRADE se nao houver um.

    Devolver o regime mais proximo sem limite de distancia parece robusto e e o
    oposto disso: o evento recebe diagnostico completo contra um historico que
    nao lhe diz respeito, e nada na resposta denuncia o encaixe.
    """
    mais_proximo = min(RPM_CONHECIDOS, key=lambda alvo: abs(alvo - rpm))
    if abs(mais_proximo - rpm) > _tolerancia_regime(mais_proximo):
        return REGIME_FORA_DA_GRADE
    return f"{int(mais_proximo)}rpm"


def regime_aproximado(rpm: float) -> bool:
    """A rotacao caiu num regime conhecido, mas nao e exatamente a dele.

    Dentro da tolerancia a comparacao continua valendo, e o aviso precisa chegar
    a quem consome pela API ou pela fila - nao so a quem esta olhando a tela.
    """
    return rpm_regime(rpm) != REGIME_FORA_DA_GRADE and float(rpm) not in RPM_CONHECIDOS


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
    # `closing` fecha o descritor e o segundo gerenciador confirma a transacao.
    # Usar so `with sqlite3.connect(...)` confirma sem fechar, e no Windows o
    # descritor aberto mantem lock no arquivo: o passo seguinte do pipeline
    # encontra o banco travado. Mesmo contrato de
    # integracao/repositorio.py::_conectar.
    with closing(sqlite3.connect(database)) as conn, conn:
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
