"""Etapa 2 dos dados: constroi e persiste o indice de similaridade."""

from __future__ import annotations

import pandas as pd

from prescritiva.config import load_settings
from prescritiva.features.build import FEATURE_COLUMNS
from prescritiva.similarity.index import IndiceSimilaridade


def main() -> None:
    settings = load_settings()
    settings.paths.ensure()
    cfg = settings.similarity

    eventos = pd.read_parquet(settings.paths.processed_dir / "eventos.parquet")
    indice = IndiceSimilaridade(cfg["n_neighbors"], cfg["min_consensus"]).fit(eventos)
    limiares = indice.calibrar_limiares(eventos, cfg["reject_distance_percentile"])
    indice.salvar(settings.paths.index_dir)

    print(f"eventos indexados: {len(eventos):,}")
    print(f"features por evento: {len(FEATURE_COLUMNS)}")
    print(f"vizinhos consultados: {cfg['n_neighbors']}")
    print("\nlimiar de rejeicao por regime (percentil "
          f"{cfg['reject_distance_percentile']:.0f} da distancia interna):")
    for regime, limiar in sorted(limiares.items()):
        print(f"  {regime:<10} {limiar:.3f}")
    print(f"\nindice salvo em {settings.paths.index_dir / 'indice_similaridade.joblib'}")


if __name__ == "__main__":
    main()
