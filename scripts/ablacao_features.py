"""Ablacao de features, decidida pela particao temporal.

Duas colunas viraram suspeitas na analise exploratoria e este script mede o
efeito de remove-las, em vez de assumir.

frequencia de pico  tem apenas 14 e 19 valores distintos, com 61 Hz respondendo
                    por 61% das linhas, inclusive nos 347 registros de motor
                    desligado. Vibracao rotacional nao existe com o motor parado,
                    entao 61 Hz e a frequencia da rede eletrica, nao uma medicao
                    do defeito. Como o RobustScaler divide por um IQR de 1.25, um
                    deslocamento de 73 Hz entre coletas vira 58 unidades de
                    distancia e domina o vizinho.

temperatura         correlaciona com a posicao dentro do bloco de gravacao entre
                    0.50 e 0.93, subindo em alguns blocos e caindo em outros, o
                    que e aquecimento e resfriamento ambiente, nao fisica de
                    falha. Separa mal: a variacao entre defeitos (1.18 C) e menor
                    que a variacao dentro de cada defeito (1.63 C).
"""

from __future__ import annotations

import pandas as pd

from prescritiva.config import load_settings
from prescritiva.evaluation.splits import amostrar, split_temporal
from prescritiva.features.build import ALL_COLUMNS, FEATURE_COLUMNS
from prescritiva.similarity.index import IndiceSimilaridade

AMOSTRA = 1500

FREQ_BRUTAS = ("z_peak_vel_comp_freq_hz", "x_peak_vel_comp_freq_hz")
ORDENS = ("ordem_z", "ordem_x")
TEMPERATURA = ("temperature_c",)


def _sem(*excluidas: tuple[str, ...]) -> tuple[str, ...]:
    fora = {coluna for grupo in excluidas for coluna in grupo}
    return tuple(c for c in ALL_COLUMNS if c not in fora)


VARIANTES: dict[str, tuple[str, ...]] = {
    "A. tudo": ALL_COLUMNS,
    "B. sem freq bruta": _sem(FREQ_BRUTAS),
    "C. sem temperatura": _sem(TEMPERATURA),
    "D. sem freq bruta e temperatura": _sem(FREQ_BRUTAS, TEMPERATURA),
    "E. producao (sem freq, temp e ordem)": FEATURE_COLUMNS,
}


def avaliar(treino: pd.DataFrame, teste: pd.DataFrame, colunas: tuple[str, ...], cfg: dict) -> dict:
    indice = IndiceSimilaridade(cfg["n_neighbors"], cfg["min_consensus"], colunas).fit(treino)
    indice.calibrar_limiares(treino, cfg["reject_distance_percentile"], amostra=800)

    acertos = rejeitados = 0
    for posicao in range(len(teste)):
        evento = teste.iloc[[posicao]]
        resultado = indice.consultar(evento)
        if not resultado.reconhecido:
            rejeitados += 1
        elif resultado.fault_predominante == evento["fault"].iloc[0]:
            acertos += 1
    reconhecidos = len(teste) - rejeitados
    return {
        "n_features": len(colunas),
        "acuracia_reconhecidos": acertos / reconhecidos if reconhecidos else 0.0,
        "taxa_rejeicao": rejeitados / len(teste),
        "acuracia_global": acertos / len(teste),
    }


def main() -> None:
    settings = load_settings()
    eventos = pd.read_parquet(settings.paths.processed_dir / "eventos.parquet")
    treino, teste_todo = split_temporal(eventos)
    teste = amostrar(teste_todo, AMOSTRA)
    print(f"particao temporal | treino {len(treino):,} | teste avaliado {len(teste):,}\n")

    print(f"{'variante':<34} {'feats':>6} {'acur.reconh':>12} {'rejeicao':>10} {'acur.global':>12}")
    print("-" * 78)
    for nome, colunas in VARIANTES.items():
        m = avaliar(treino, teste, colunas, settings.similarity)
        print(
            f"{nome:<34} {m['n_features']:>6} {m['acuracia_reconhecidos']:>12.3f} "
            f"{m['taxa_rejeicao']:>10.3f} {m['acuracia_global']:>12.3f}"
        )


if __name__ == "__main__":
    main()
