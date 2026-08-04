"""Analise exploratoria que valida cada decisao de tratamento antes de aplica-la.

Quatro hipoteses. Duas viraram regra de ingestao, duas descartaram features.
Nenhuma foi assumida: todas estao medidas aqui, e o resultado de cada uma esta
citado no comentario do codigo que a aplica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prescritiva.config import load_settings
from prescritiva.data.ingest import load_raw, normalize_fault, rpm_regime
from prescritiva.data.schema import SENSOR_COLUMNS, UNIT_DUPLICATES


def h1_unidades_redundantes(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("H1 - as colunas imperiais sao redundantes com as metricas?")
    print("=" * 78)
    print(f"{'par':<52} {'correlacao':>11} {'fator':>9}")
    for imperial, metrica in UNIT_DUPLICATES.items():
        validos = df[df[imperial].abs() > 1e-9]
        print(
            f"{imperial + ' ~ ' + metrica:<52} {df[imperial].corr(df[metrica]):>11.6f} "
            f"{(validos[metrica] / validos[imperial]).median():>9.3f}"
        )
    print("\nSIM. Fator 25.4 = polegada para milimetro; a temperatura e conversao afim,")
    print("por isso correlacao 1.0 com fator nao constante. Manter os dois lados daria")
    print("peso duplo a mesma medida na distancia -> descartar as imperiais.")


def h2_campanhas_de_coleta(df: pd.DataFrame, padrao: str) -> None:
    print("\n" + "=" * 78)
    print('H2 - o sufixo "_2"/"_3" e campanha de coleta do mesmo defeito?')
    print("=" * 78)
    janelas = df.groupby("fault")["created_at"].agg(["min", "max", "count"]).sort_values("min")
    print(janelas.to_string())
    sobrepostos = 0
    anterior_fim = None
    for _, linha in janelas.iterrows():
        if anterior_fim is not None and linha["min"] < anterior_fim:
            sobrepostos += 1
        anterior_fim = max(anterior_fim or linha["max"], linha["max"])
    print(f"\nblocos com sobreposicao temporal: {sobrepostos} de {len(janelas)}")
    print("\nSIM, pelo tempo. Cada rotulo ocupa um bloco continuo e a segunda rodada de")
    print("campanhas roda inteira no fim do periodo. O rotulo base passa a valer para")
    print("diagnostico e cobertura documental; o original fica preservado porque a")
    print("campanha e o grupo de vazamento que a avaliacao precisa separar.")

    base = df["fault"].map(lambda f: normalize_fault(f, padrao))
    print(f"\nrotulos antes: {df['fault'].nunique()} | depois: {base.nunique()}")


def h3_frequencia_de_pico(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("H3 - a frequencia de pico mede o defeito?")
    print("=" * 78)
    for coluna in ("z_peak_vel_comp_freq_hz", "x_peak_vel_comp_freq_hz"):
        contagem = df[coluna].value_counts()
        print(
            f"{coluna:<26} valores distintos: {df[coluna].nunique():>3} | "
            f"mais comum: {contagem.index[0]:>6} Hz em {contagem.iloc[0] / len(df):>5.1%} das linhas"
        )
    parado = df[df["rpm"] == 0]
    print(f"\ncom o motor parado ({len(parado)} registros), a frequencia de pico e "
          f"{parado['z_peak_vel_comp_freq_hz'].median():.1f} Hz")
    print("\nNAO. Vibracao rotacional nao existe com o motor parado, entao 61 Hz e a")
    print("frequencia da rede eletrica, nao uma medicao do defeito. Como o RobustScaler")
    print("divide por um IQR de 1.25, um deslocamento entre coletas vira dezenas de")
    print("unidades de distancia e domina o vizinho -> descartar do espaco de busca.")


def h4_temperatura(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("H4 - a temperatura e sintoma ou relogio?")
    print("=" * 78)
    correlacoes = []
    for bloco, grupo in df.groupby("fault"):
        ordenado = grupo.sort_values("created_at").reset_index(drop=True)
        if len(ordenado) < 100:
            continue
        correlacoes.append(
            (bloco, abs(ordenado["temperature_c"].corr(pd.Series(range(len(ordenado))))))
        )
    correlacoes.sort(key=lambda item: -item[1])
    print(f"{'bloco':<28} {'|corr| com a posicao no tempo':>30}")
    for bloco, corr in correlacoes[:8]:
        print(f"{bloco:<28} {corr:>30.3f}")

    entre = df.groupby("fault")["temperature_c"].median().std()
    dentro = df.groupby("fault")["temperature_c"].std().mean()
    print(f"\ndesvio entre medianas de defeito:   {entre:.2f} C")
    print(f"desvio medio dentro de cada defeito: {dentro:.2f} C")
    print(f"razao entre/dentro: {entre / dentro:.2f}")
    print("\nRELOGIO. A correlacao com a posicao no bloco e alta e troca de sinal entre")
    print("blocos, o que e aquecimento e resfriamento ambiente, nao fisica de falha. E a")
    print("razao entre/dentro abaixo de 1 diz que a temperatura separa os defeitos pior")
    print("do que varia dentro de cada um -> descartar do espaco de busca.")


def resumo(df: pd.DataFrame, padrao: str, estados: list[str]) -> None:
    print("\n" + "=" * 78)
    print("Distribuicao final apos normalizacao de rotulo")
    print("=" * 78)
    base = df["fault"].map(lambda f: normalize_fault(f, padrao))
    tabela = (
        pd.DataFrame({"defeito": base})
        .value_counts()
        .rename("registros")
        .reset_index()
        .assign(
            tipo=lambda d: np.where(d["defeito"].isin(estados), "condicao de operacao", "defeito"),
            pct=lambda d: (d["registros"] / len(df) * 100).round(1),
        )
    )
    print(tabela.to_string(index=False))
    print(f"\ntotal: {len(df):,} registros | {tabela['defeito'].nunique()} rotulos")


def main() -> None:
    settings = load_settings()
    df = load_raw(settings.paths.raw_csv)
    df["regime"] = df["rpm"].map(rpm_regime)
    padrao = settings.ingest["campaign_suffix_pattern"]

    h1_unidades_redundantes(df)
    h2_campanhas_de_coleta(df, padrao)
    h3_frequencia_de_pico(df)
    h4_temperatura(df)
    resumo(df, padrao, settings.ingest["operational_states"])


if __name__ == "__main__":
    main()
