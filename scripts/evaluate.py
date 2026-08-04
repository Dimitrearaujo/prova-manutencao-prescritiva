"""Avaliacao do motor de similaridade.

Quatro experimentos, cada um respondendo a uma pergunta diferente:

aleatorio  Nao e a avaliacao da solucao. Esta aqui para mostrar por que ele nao
           vale neste conjunto de dados: o ensaio gravou cada defeito em um bloco
           continuo de tempo, e sortear coloca leituras a segundos de distancia
           nos dois lados da particao.

temporal   A metrica principal. Dentro de cada bloco, treino no inicio e teste no
           fim, com intervalo de guarda no meio. Representa o uso real:
           reconhecer um defeito ja visto a partir de uma leitura posterior.

campanha   Teste de estresse. Indexa a primeira campanha e consulta as seguintes,
           coletadas depois. Mede o que acontece sob recalibracao da
           instrumentacao.

inedito    Remove uma classe inteira do indice e pergunta por ela. Mede a unica
           coisa que separa esta solucao de um classificador fechado: dizer "nao
           reconheco" em vez de escolher o rotulo menos errado.

Cada um mede em dois niveis. O acerto de ROTULO cobra o nome exato do defeito.
O acerto de PROCEDIMENTO cobra o que o sistema entrega ao tecnico: os quatro
defeitos de rolamento levam ao mesmo Doc1 e a mesma acao corretiva, entao trocar
anel interno por anel externo nao muda uma linha da instrucao. O segundo numero e
o que mede o produto; o primeiro e um detalhe de nomenclatura.
"""

from __future__ import annotations

import json

import pandas as pd

from prescritiva.config import load_catalog, load_settings
from prescritiva.evaluation.splits import amostrar, split_aleatorio, split_campanha, split_temporal
from prescritiva.similarity.index import IndiceSimilaridade

AMOSTRA_TESTE = 2000
AMOSTRA_INEDITO = 300


def _mapa_procedimento() -> dict[str, str]:
    """Rotulo -> acao prescrita. Rotulos que levam ao mesmo procedimento colapsam."""
    catalogo = load_catalog()
    mapa = {fault: e["componente"] for fault, e in catalogo["faults"].items()}
    mapa.update({estado: "condicao_operacional" for estado in catalogo["estados_operacionais"]})
    return mapa


def _consultar_muitos(indice: IndiceSimilaridade, teste: pd.DataFrame) -> pd.DataFrame:
    mapa = _mapa_procedimento()
    linhas = []
    for posicao in range(len(teste)):
        evento = teste.iloc[[posicao]]
        resultado = indice.consultar(evento)
        verdadeiro = evento["fault"].iloc[0]
        previsto = resultado.fault_predominante
        linhas.append(
            {
                "verdadeiro": verdadeiro,
                "previsto": previsto,
                "proc_verdadeiro": mapa.get(verdadeiro, verdadeiro),
                "proc_previsto": mapa.get(previsto, previsto),
                "confianca": resultado.confianca,
                "distancia": resultado.distancia_media,
                "reconhecido": resultado.reconhecido,
            }
        )
    return pd.DataFrame(linhas)


def _metricas(res: pd.DataFrame) -> dict:
    reconhecidos = res[res["reconhecido"]]
    if not len(reconhecidos):
        return {"eventos": len(res), "taxa_rejeicao": 1.0, "acuracia_rotulo": 0.0,
                "acuracia_procedimento": 0.0, "acuracia_global": 0.0,
                "distancia_media": float(res["distancia"].mean())}
    return {
        "eventos": len(res),
        "taxa_rejeicao": float((~res["reconhecido"]).mean()),
        "acuracia_rotulo": float((reconhecidos["verdadeiro"] == reconhecidos["previsto"]).mean()),
        "acuracia_procedimento": float(
            (reconhecidos["proc_verdadeiro"] == reconhecidos["proc_previsto"]).mean()
        ),
        "acuracia_global": float((res["verdadeiro"] == res["previsto"]).mean()),
        "distancia_media": float(res["distancia"].mean()),
    }


def _rodar(treino: pd.DataFrame, teste: pd.DataFrame, cfg: dict, *, amostra_calibracao: int = 1500):
    indice = IndiceSimilaridade(cfg["n_neighbors"], cfg["min_consensus"]).fit(treino)
    indice.calibrar_limiares(treino, cfg["reject_distance_percentile"], amostra=amostra_calibracao)
    return _consultar_muitos(indice, teste)


def experimento_defeito_inedito(eventos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    linhas = []
    for alvo in sorted(eventos.loc[eventos["e_defeito"], "fault"].unique()):
        treino = eventos[eventos["fault"] != alvo]
        teste = amostrar(eventos[eventos["fault"] == alvo], AMOSTRA_INEDITO)
        resultado = _rodar(treino, teste, cfg, amostra_calibracao=600)
        reconhecidos = resultado[resultado["reconhecido"]]
        linhas.append(
            {
                "defeito_removido": alvo,
                "eventos": len(resultado),
                "rejeitado_corretamente": float((~resultado["reconhecido"]).mean()),
                "rotulo_errado_mais_dado": reconhecidos["previsto"].value_counts().idxmax()
                if len(reconhecidos)
                else "-",
            }
        )
    return pd.DataFrame(linhas)


def main() -> None:
    settings = load_settings()
    cfg = settings.similarity
    eventos = pd.read_parquet(settings.paths.processed_dir / "eventos.parquet")

    particoes = {
        "Aleatorio (invalido)": split_aleatorio(eventos),
        "Temporal (principal)": split_temporal(eventos),
        "Nova campanha (estresse)": split_campanha(eventos),
    }

    splits: dict[str, dict] = {}
    matrizes: dict[str, dict] = {}
    for nome, (treino, teste_todo) in particoes.items():
        teste = amostrar(teste_todo, AMOSTRA_TESTE)
        resultado = _rodar(treino, teste, cfg)
        splits[nome] = _metricas(resultado)
        reconhecidos = resultado[resultado["reconhecido"]]
        matrizes[nome] = (
            pd.crosstab(reconhecidos["verdadeiro"], reconhecidos["previsto"], normalize="index")
            .round(2)
            .to_dict()
            if len(reconhecidos)
            else {}
        )

        print("=" * 78)
        print(nome.upper())
        print("=" * 78)
        for chave, valor in splits[nome].items():
            print(f"  {chave:<32} {valor:.4f}" if isinstance(valor, float) else f"  {chave:<32} {valor}")
        print()

    print("=" * 78)
    print("DEFEITO INEDITO (a classe some do indice e e consultada)")
    print("=" * 78)
    inedito = experimento_defeito_inedito(eventos, cfg)
    print(inedito.to_string(index=False))
    print(f"\n  rejeicao media: {inedito['rejeitado_corretamente'].mean():.1%}")

    destino = settings.paths.index_dir / "avaliacao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                "splits": splits,
                "matrizes_confusao": matrizes,
                "defeito_inedito": inedito.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nresultados salvos em {destino}")


if __name__ == "__main__":
    main()
