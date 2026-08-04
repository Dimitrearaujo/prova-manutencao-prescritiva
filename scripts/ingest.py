"""Etapa 1 dos dados: banner.csv -> parquet analitico + SQLite indexado."""

from __future__ import annotations

from prescritiva.config import load_settings
from prescritiva.data.ingest import run


def main() -> None:
    settings = load_settings()
    df = run()
    print(f"registros ingeridos: {len(df):,}")
    print(f"rotulos apos normalizacao: {df['fault'].nunique()}")
    print(f"defeitos: {int(df['e_defeito'].sum()):,} | estados operacionais: {int((~df['e_defeito']).sum()):,}")
    print("\nregistros por regime de rotacao:")
    print(df["regime_rpm"].value_counts().to_string())
    print("\nregistros por campanha:")
    print(df["campanha"].value_counts().sort_index().to_string())
    print(f"\nparquet: {settings.paths.processed_dir / 'eventos.parquet'}")
    print(f"sqlite:  {settings.paths.database}")


if __name__ == "__main__":
    main()
