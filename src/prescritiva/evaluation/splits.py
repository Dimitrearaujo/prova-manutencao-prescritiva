"""Particoes de treino e teste.

O ensaio gravou um defeito de cada vez, em blocos continuos de tempo. Isso torna
o sorteio aleatorio invalido: leituras a segundos de distancia caem nos dois
lados e a mesma medicao aparece no treino e no teste com outro carimbo de hora.
Todas as particoes honestas deste projeto respeitam o tempo.
"""

from __future__ import annotations

import pandas as pd


def split_aleatorio(eventos: pd.DataFrame, frac_treino: float = 0.8, semente: int = 42):
    """Vazado de proposito. Serve de contraste, nunca de metrica."""
    embaralhado = eventos.sample(frac=1.0, random_state=semente)
    corte = int(len(embaralhado) * frac_treino)
    return embaralhado.iloc[:corte], embaralhado.iloc[corte:]


def split_temporal(
    eventos: pd.DataFrame, frac_treino: float = 0.6, frac_guarda: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dentro de cada bloco de gravacao, treino no inicio e teste no fim.

    O intervalo de guarda descartado no meio existe porque leituras vizinhas no
    tempo sao quase identicas: sem ele, a ultima linha do treino e a primeira do
    teste sao a mesma medicao.

    E a particao que representa o uso real: reconhecer um defeito ja visto, a
    partir de uma leitura posterior da mesma operacao.
    """
    treino, teste = [], []
    for _, bloco in eventos.groupby("fault_original"):
        ordenado = bloco.sort_values("created_at")
        fim_treino = int(len(ordenado) * frac_treino)
        inicio_teste = int(len(ordenado) * (frac_treino + frac_guarda))
        treino.append(ordenado.iloc[:fim_treino])
        teste.append(ordenado.iloc[inicio_teste:])
    return pd.concat(treino), pd.concat(teste)


def split_campanha(eventos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Indexa a primeira campanha e consulta as seguintes.

    E um teste de estresse, nao a metrica principal: mede o que acontece quando a
    instrumentacao e recalibrada entre coletas. O teste fica restrito aos
    defeitos que a primeira campanha conhece.
    """
    treino = eventos[eventos["campanha"] == 1]
    teste = eventos[(eventos["campanha"] > 1) & eventos["fault"].isin(treino["fault"].unique())]
    return treino, teste


def amostrar(df: pd.DataFrame, n: int, semente: int = 42) -> pd.DataFrame:
    """Amostra estratificada por rotulo, preservando a proporcao original."""
    if len(df) <= n:
        return df
    partes = [
        grupo.sample(n=max(1, round(n * len(grupo) / len(df))), random_state=semente)
        for _, grupo in df.groupby("fault")
    ]
    return pd.concat(partes).sample(frac=1.0, random_state=semente)
