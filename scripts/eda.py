"""Analise exploratoria que valida cada decisao de tratamento antes de aplica-la.

Cinco hipoteses. Duas viraram regra de ingestao, duas descartaram features e uma
mede a deriva de linha de base que a avaliacao por campanha encontra depois.
Nenhuma foi assumida: todas estao medidas aqui, e o resultado de cada uma esta
citado no comentario do codigo que a aplica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from prescritiva.config import load_settings
from prescritiva.data.ingest import load_raw, normalize_fault, rpm_regime
from prescritiva.data.schema import SENSOR_COLUMNS, UNIT_DUPLICATES
from prescritiva.features.build import FEATURE_COLUMNS, build_features

FREQUENCIAS_DE_PICO: tuple[str, ...] = ("z_peak_vel_comp_freq_hz", "x_peak_vel_comp_freq_hz")


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


def h2_campanhas_de_coleta(df: pd.DataFrame) -> None:
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

    # A leitura comoda seria "a segunda rodada roda inteira no fim do periodo".
    # Ela e falsa e a avaliacao por campanha depende dela, entao o desalinhamento
    # entre as duas rodadas e medido aqui em vez de ficar implicito no grafico.
    inicio_segunda = df.loc[df["campanha"] > 1, "created_at"].min()
    tardios = df[(df["campanha"] == 1) & (df["created_at"] > inicio_segunda)]
    print(f"primeiro evento da segunda rodada: {inicio_segunda}")
    print(f"eventos da primeira rodada gravados depois disso: {len(tardios):,}")
    if len(tardios):
        print(tardios["fault"].value_counts().to_string())

    desalinhados = " e ".join(sorted(tardios["fault"].unique()))
    print("\nSIM, pelo tempo. O sufixo marca uma nova gravacao do mesmo defeito, e a segunda")
    print("rodada concentra-se no fim do periodo. Nao e uma linha limpa: os blocos se")
    print(f"sobrepoem, e parte da primeira rodada de {desalinhados}")
    print("foi gravada ja durante a segunda, o que enfraquece a leitura de deriva temporal")
    print("para essas classes. O rotulo base passa a valer para diagnostico e cobertura")
    print("documental; o original fica preservado porque a campanha e o grupo de vazamento")
    print("que a avaliacao precisa separar.")

    print(f"\nrotulos antes: {df['fault'].nunique()} | depois: {df['base'].nunique()}")


def _iqr_por_regime(df: pd.DataFrame, colunas: tuple[str, ...]) -> pd.DataFrame:
    """IQR com que o RobustScaler divide cada coluna, medido por regime.

    O indice ajusta um scaler por regime de rotacao (similarity/index.py::fit),
    entao um IQR global nao descreve nenhuma das buscas que o sistema faz de
    fato. Reproduzir o mesmo agrupamento e o que torna o numero verificavel.
    """
    linhas: dict[str, dict[str, float]] = {}
    for regime, grupo in df.groupby("regime"):
        scaler = RobustScaler().fit(build_features(grupo, colunas))
        linhas[regime] = dict(zip(colunas, scaler.scale_, strict=True))
    return pd.DataFrame(linhas).T


def _distancia_tipica(df: pd.DataFrame, amostra: int = 600) -> dict[str, float]:
    """Distancia media aos 25 vizinhos no espaco de busca de producao, por regime.

    Serve de regua: dizer que uma coluna vale dezenas de unidades de distancia so
    significa alguma coisa ao lado da distancia que separa dois eventos que o
    sistema considera parecidos.
    """
    tipicas: dict[str, float] = {}
    for regime, grupo in df.groupby("regime"):
        recorte = grupo.sample(n=min(amostra, len(grupo)), random_state=42)
        matriz = RobustScaler().fit_transform(build_features(recorte, FEATURE_COLUMNS))
        vizinhos = min(26, len(matriz))
        distancias, _ = NearestNeighbors(n_neighbors=vizinhos).fit(matriz).kneighbors(matriz)
        tipicas[regime] = float(np.median(distancias[:, 1:].mean(axis=1)))
    return tipicas


def h3_frequencia_de_pico(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("H3 - a frequencia de pico mede o defeito?")
    print("=" * 78)
    for coluna in FREQUENCIAS_DE_PICO:
        contagem = df[coluna].value_counts()
        print(
            f"{coluna:<26} valores distintos: {df[coluna].nunique():>3} | "
            f"mais comum: {contagem.index[0]:>6} Hz em {contagem.iloc[0] / len(df):>5.1%} das linhas"
        )
    parado = df[df["rpm"] == 0]
    print(f"\ncom o motor parado ({len(parado)} registros), a frequencia de pico e "
          f"{parado['z_peak_vel_comp_freq_hz'].median():.1f} Hz")

    iqrs = _iqr_por_regime(df, FREQUENCIAS_DE_PICO)
    tipicas = _distancia_tipica(df)
    print("\nescala da coluna no espaco de busca, por regime")
    print(f"{'regime':<9} {'coluna':<26} {'IQR':>7} {'amplitude':>11} {'em unidades':>12} "
          f"{'dist. tipica':>13}")
    unidades: list[float] = []
    for regime, grupo in df.groupby("regime"):
        for coluna in FREQUENCIAS_DE_PICO:
            iqr = float(iqrs.loc[regime, coluna])
            amplitude = float(grupo[coluna].max() - grupo[coluna].min())
            unidades.append(amplitude / iqr)
            print(f"{regime:<9} {coluna:<26} {iqr:>7.2f} {amplitude:>8.1f} Hz "
                  f"{amplitude / iqr:>12.1f} {tipicas[regime]:>13.2f}")

    print("\nNAO. Vibracao rotacional nao existe com o motor parado, entao 61 Hz e a")
    print("frequencia da rede eletrica, nao uma medicao do defeito. E o IQR que divide")
    print(f"essa coluna varia de {iqrs.min().min():.1f} a {iqrs.max().max():.1f} conforme "
          "o regime: coluna que mede defeito")
    print("nao muda de escala assim. Onde o IQR e pequeno a amplitude da coluna vale ate")
    print(f"{max(unidades):.0f} unidades de distancia, contra as "
          f"{np.median(list(tipicas.values())):.1f} que separam dois eventos parecidos:")
    print("um deslocamento entre coletas domina a busca sozinho -> descartar do espaco.")


def h4_temperatura(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("H4 - a temperatura e sintoma ou relogio?")
    print("=" * 78)
    # A correlacao com o tempo e medida por BLOCO DE GRAVACAO (rotulo bruto, com
    # o sufixo de campanha) porque o que aquece e esfria e a sessao de gravacao:
    # juntar duas sessoes do mesmo defeito somaria dois aquecimentos separados.
    correlacoes: list[tuple[str, float, bool]] = []
    for bloco, grupo in df.groupby("fault"):
        ordenado = grupo.sort_values("created_at").reset_index(drop=True)
        if len(ordenado) < 100:
            continue
        correlacao = ordenado["temperature_c"].corr(pd.Series(range(len(ordenado))))
        correlacoes.append((bloco, float(correlacao), ordenado["base"].iloc[0] != bloco))
    correlacoes.sort(key=lambda item: -abs(item[1]))

    # A lista sai inteira de proposito. Cortada nos primeiros, ela mostra so
    # correlacao alta e sugere um piso que boa parte dos blocos nao alcanca.
    print(f"{'bloco de gravacao':<28} {'corr com a posicao no tempo':>30} {'rodada':>8}")
    for bloco, correlacao, segunda in correlacoes:
        print(f"{bloco:<28} {correlacao:>30.3f} {('2a' if segunda else '1a'):>8}")

    modulos = [abs(correlacao) for _, correlacao, _ in correlacoes]
    fortes = [segunda for _, correlacao, segunda in correlacoes if abs(correlacao) >= 0.80]
    print(f"\nfaixa observada: |corr| de {min(modulos):.3f} a {max(modulos):.3f} em "
          f"{len(correlacoes)} blocos, {sum(1 for m in modulos if m < 0.50)} abaixo de 0.50")
    print(f"acima de 0.80: {len(fortes)} blocos, {sum(fortes)} deles da segunda rodada")

    # A separabilidade e medida por DEFEITO (rotulo normalizado), nao por bloco:
    # o espaco de busca compara defeitos, e "cocked_rotor" e "cocked_rotor_2" sao
    # o mesmo defeito. Medida pelo rotulo bruto a razao sai 1.09 e contradiz a
    # conclusao, porque a diferenca entre duas campanhas do MESMO defeito entra
    # na conta como se fosse separacao entre defeitos diferentes.
    entre = df.groupby("base")["temperature_c"].median().std()
    dentro = df.groupby("base")["temperature_c"].std().mean()
    razao = entre / dentro
    print(f"\ndesvio entre medianas de defeito:    {entre:.2f} C")
    print(f"desvio medio dentro de cada defeito: {dentro:.2f} C")
    print(f"razao entre/dentro: {razao:.2f}")
    print("\nRELOGIO. A correlacao com a posicao no bloco troca de sinal entre blocos e os")
    print("mais fortes sao todos da segunda rodada de gravacao, o que e aquecimento e")
    print("resfriamento ambiente da sessao, nao fisica de falha. E a razao entre/dentro")
    print(f"de {razao:.2f}, abaixo de 1, diz que a temperatura separa os defeitos pior do que")
    print("varia dentro de cada um -> descartar do espaco de busca.")


def h5_deriva_entre_campanhas(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("H5 - a linha de base se deslocou entre as campanhas de coleta?")
    print("=" * 78)
    # Restrito a condicao normal de proposito: se a maquina sem defeito le
    # diferente na segunda rodada, a diferenca vem da instrumentacao ou do
    # ambiente, e nao do defeito. E a unica forma de separar deriva de sintoma.
    normal = df[df["base"] == "normal"]
    primeira = normal[normal["campanha"] == 1]
    segunda = normal[normal["campanha"] > 1]
    print(f"condicao normal | primeira rodada {len(primeira):,} eventos | "
          f"segunda rodada {len(segunda):,} eventos")

    variacoes: list[tuple[str, float, float, float]] = []
    for coluna in SENSOR_COLUMNS:
        m1, m2 = primeira[coluna].median(), segunda[coluna].median()
        if abs(m1) > 1e-9:
            variacoes.append((coluna, m1, m2, (m2 / m1 - 1) * 100))
    variacoes.sort(key=lambda item: -abs(item[3]))
    print(f"\n{'coluna':<26} {'mediana 1a':>11} {'mediana 2a':>11} {'variacao':>10}")
    for coluna, m1, m2, variacao in variacoes[:6]:
        print(f"{coluna:<26} {m1:>11.4f} {m2:>11.4f} {variacao:>9.1f}%")

    # O agregado esconde o regime: a mesma coluna anda muito a 2000 rpm e quase
    # nada a 500. Publicar so o maior recorte seria escolher o numero.
    print("\nz_rms_acceleration_g por rotacao")
    print(f"{'rpm':<9} {'mediana 1a':>11} {'mediana 2a':>11} {'variacao':>10}")
    por_rpm: dict[float, float] = {}
    for rpm, grupo in normal.groupby("rpm"):
        m1 = grupo[grupo["campanha"] == 1]["z_rms_acceleration_g"].median()
        m2 = grupo[grupo["campanha"] > 1]["z_rms_acceleration_g"].median()
        if pd.isna(m1) or pd.isna(m2) or abs(m1) <= 1e-9:
            continue
        por_rpm[float(rpm)] = (m2 / m1 - 1) * 100
        print(f"{rpm:<9.0f} {m1:>11.4f} {m2:>11.4f} {por_rpm[float(rpm)]:>9.1f}%")

    agregado = next(v for c, _, _, v in variacoes if c == "z_rms_acceleration_g")
    pior_rpm = max(por_rpm, key=lambda rpm: por_rpm[rpm])
    print("\nSIM. A maquina sem defeito le diferente nas duas rodadas: a aceleracao RMS no")
    print(f"eixo Z sobe {agregado:.1f}% na mediana agregada e {por_rpm[pior_rpm]:.1f}% "
          f"a {pior_rpm:.0f} rpm. O deslocamento e")
    print("da linha de base, nao do defeito, e e a causa fisica da queda de desempenho")
    print("que scripts/evaluate.py mede na particao por campanha -> justifica monitorar")
    print("deriva pela taxa de rejeicao em vez de confiar no indice indefinidamente.")


def resumo(df: pd.DataFrame, estados: list[str]) -> None:
    print("\n" + "=" * 78)
    print("Distribuicao final apos normalizacao de rotulo")
    print("=" * 78)
    tabela = (
        pd.DataFrame({"defeito": df["base"]})
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
    padrao = settings.ingest["campaign_suffix_pattern"]
    df["regime"] = df["rpm"].map(rpm_regime)
    # Mesmas derivacoes de data/ingest.py::prepare, repetidas aqui porque prepare
    # ja descarta as colunas imperiais que H1 precisa comparar.
    df["base"] = df["fault"].map(lambda f: normalize_fault(f, padrao))
    df["campanha"] = df["fault"].str.extract(padrao, expand=False).fillna("1").astype(int)

    h1_unidades_redundantes(df)
    h2_campanhas_de_coleta(df)
    h3_frequencia_de_pico(df)
    h4_temperatura(df)
    h5_deriva_entre_campanhas(df)
    resumo(df, settings.ingest["operational_states"])


if __name__ == "__main__":
    main()
