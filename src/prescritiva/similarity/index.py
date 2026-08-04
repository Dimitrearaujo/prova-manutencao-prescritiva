"""Busca por similaridade no historico operacional.

O enunciado e explicito: a solucao nao deve depender de classificacao previa de
falhas conhecidas, e sim identificar padroes semelhantes dentro do historico.
Por isso aqui nao existe classificador treinado com rotulo como alvo. Existe um
indice de vizinhos: o evento novo e comparado ao passado, os vizinhos trazem
seus proprios rotulos, e o diagnostico e o consenso deles.

A consequencia pratica e que um defeito novo, nunca visto, nao e forcado dentro
de uma classe existente: ele aparece como distancia alta e cai na regra de
rejeicao.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from prescritiva.features.build import FEATURE_COLUMNS, build_features


@dataclass
class Vizinho:
    id: int
    fault: str
    fault_original: str
    created_at: str
    rpm: float
    distancia: float


@dataclass
class ResultadoSimilaridade:
    regime_rpm: str
    vizinhos: list[Vizinho]
    distribuicao: dict[str, float]
    fault_predominante: str
    confianca: float
    distancia_media: float
    limiar_rejeicao: float
    reconhecido: bool
    motivo: str = ""


class IndiceSimilaridade:
    """Um indice de vizinhos por regime de rotacao.

    Manter indices separados por rpm e mais barato e mais correto do que um
    indice unico com o rpm como feature: evita que a distancia seja dominada
    pela rotacao e garante que todo vizinho retornado seja comparavel.
    """

    def __init__(
        self,
        n_neighbors: int = 25,
        min_consensus: float = 0.45,
        colunas: tuple[str, ...] = FEATURE_COLUMNS,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_consensus = min_consensus
        self.colunas = tuple(colunas)
        self._scalers: dict[str, RobustScaler] = {}
        self._indices: dict[str, NearestNeighbors] = {}
        self._metadados: dict[str, pd.DataFrame] = {}
        self._limiares: dict[str, float] = {}

    def fit(self, eventos: pd.DataFrame, *, limiares: dict[str, float] | None = None) -> "IndiceSimilaridade":
        for regime, grupo in eventos.groupby("regime_rpm"):
            features = build_features(grupo, self.colunas)
            scaler = RobustScaler().fit(features)
            matriz = scaler.transform(features)

            vizinhos = min(self.n_neighbors, len(grupo))
            # Forca bruta e a escolha certa aqui: em 25 dimensoes as arvores de
            # particionamento degradam e ficam mais lentas que a varredura.
            indice = NearestNeighbors(
                n_neighbors=vizinhos, metric="euclidean", algorithm="brute", n_jobs=-1
            ).fit(matriz)

            self._scalers[regime] = scaler
            self._indices[regime] = indice
            self._metadados[regime] = grupo[
                ["id", "fault", "fault_original", "created_at", "rpm", "campanha"]
            ].reset_index(drop=True)
        self._limiares = limiares or {}
        return self

    def calibrar_limiares(
        self, eventos: pd.DataFrame, percentil: float, *, amostra: int = 1500
    ) -> dict[str, float]:
        """Define a distancia acima da qual o padrao e considerado desconhecido.

        Calibrado sobre o proprio historico: se um evento novo esta mais longe
        do historico do que o percentil escolhido dos proprios eventos, ele nao
        se parece com nada ja visto e nao deve receber um palpite.
        """
        limiares: dict[str, float] = {}
        for regime, grupo in eventos.groupby("regime_rpm"):
            referencia = grupo.sample(n=min(amostra, len(grupo)), random_state=42)
            matriz = self._scalers[regime].transform(build_features(referencia, self.colunas))
            # O primeiro vizinho e o proprio ponto, por isso a coluna 1 em diante.
            distancias, _ = self._indices[regime].kneighbors(matriz)
            limiares[regime] = float(np.percentile(distancias[:, 1:].mean(axis=1), percentil))
        self._limiares = limiares
        return limiares

    def consultar(self, evento: pd.DataFrame, *, excluir_ids: set[int] | None = None) -> ResultadoSimilaridade:
        regime = evento["regime_rpm"].iloc[0]
        if regime not in self._indices:
            return ResultadoSimilaridade(
                regime_rpm=regime,
                vizinhos=[],
                distribuicao={},
                fault_predominante="",
                confianca=0.0,
                distancia_media=float("inf"),
                limiar_rejeicao=0.0,
                reconhecido=False,
                motivo=f"Nenhum historico no regime de {regime}.",
            )

        matriz = self._scalers[regime].transform(build_features(evento, self.colunas))
        # Pede folga para poder descartar o proprio evento quando ele veio do historico.
        k = min(self.n_neighbors + len(excluir_ids or ()), len(self._metadados[regime]))
        distancias, posicoes = self._indices[regime].kneighbors(matriz, n_neighbors=k)

        meta = self._metadados[regime]
        vizinhos: list[Vizinho] = []
        for distancia, posicao in zip(distancias[0], posicoes[0], strict=True):
            linha = meta.iloc[posicao]
            if excluir_ids and int(linha["id"]) in excluir_ids:
                continue
            vizinhos.append(
                Vizinho(
                    id=int(linha["id"]),
                    fault=str(linha["fault"]),
                    fault_original=str(linha["fault_original"]),
                    created_at=str(linha["created_at"]),
                    rpm=float(linha["rpm"]),
                    distancia=float(distancia),
                )
            )
            if len(vizinhos) == self.n_neighbors:
                break

        pesos = _pesos_por_distancia([v.distancia for v in vizinhos])
        distribuicao: dict[str, float] = {}
        for vizinho, peso in zip(vizinhos, pesos, strict=True):
            distribuicao[vizinho.fault] = distribuicao.get(vizinho.fault, 0.0) + peso
        distribuicao = dict(sorted(distribuicao.items(), key=lambda kv: kv[1], reverse=True))

        predominante = next(iter(distribuicao), "")
        confianca = distribuicao.get(predominante, 0.0)
        distancia_media = float(np.mean([v.distancia for v in vizinhos])) if vizinhos else float("inf")
        limiar = self._limiares.get(regime, float("inf"))

        reconhecido, motivo = True, ""
        if distancia_media > limiar:
            reconhecido = False
            motivo = (
                f"O evento esta a uma distancia media de {distancia_media:.2f} do historico, "
                f"acima do limiar de {limiar:.2f} calibrado para {regime}. "
                "Nenhum padrao conhecido se parece o suficiente com ele."
            )
        elif confianca < self.min_consensus:
            reconhecido = False
            motivo = (
                f"Os vizinhos nao concordam: o rotulo mais votado reune apenas "
                f"{confianca:.0%} do peso, abaixo do minimo de {self.min_consensus:.0%}."
            )

        return ResultadoSimilaridade(
            regime_rpm=regime,
            vizinhos=vizinhos,
            distribuicao=distribuicao,
            fault_predominante=predominante,
            confianca=float(confianca),
            distancia_media=distancia_media,
            limiar_rejeicao=limiar,
            reconhecido=reconhecido,
            motivo=motivo,
        )

    def salvar(self, destino: Path) -> None:
        destino.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "n_neighbors": self.n_neighbors,
                "min_consensus": self.min_consensus,
                "scalers": self._scalers,
                "indices": self._indices,
                "metadados": self._metadados,
                "limiares": self._limiares,
                "colunas": list(self.colunas),
            },
            destino / "indice_similaridade.joblib",
            compress=3,
        )

    @classmethod
    def carregar(cls, origem: Path) -> "IndiceSimilaridade":
        estado = joblib.load(origem / "indice_similaridade.joblib")
        indice = cls(estado["n_neighbors"], estado["min_consensus"], tuple(estado["colunas"]))
        indice._scalers = estado["scalers"]
        indice._indices = estado["indices"]
        indice._metadados = estado["metadados"]
        indice._limiares = estado["limiares"]
        return indice


def _pesos_por_distancia(distancias: list[float]) -> list[float]:
    """Vizinho mais proximo pesa mais, e os pesos somam 1."""
    if not distancias:
        return []
    inversos = 1.0 / (np.asarray(distancias) + 1e-6)
    return list(inversos / inversos.sum())
