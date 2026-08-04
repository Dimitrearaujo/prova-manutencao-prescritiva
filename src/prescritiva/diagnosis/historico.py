"""Estatisticas historicas do defeito identificado.

Responde as tres caixas de saida do diagrama do enunciado que nao vem do texto:
quantidade de ocorrencias, distribuicao ao longo do tempo e frequencia. As
consultas vao ao SQLite em vez de varrer o parquet em memoria porque a estacao
de operacao tem orcamento de RAM fixo e o volume historico so cresce.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EstatisticaHistorica:
    fault: str
    ocorrencias: int
    primeira: str | None
    ultima: str | None
    dias_cobertos: int
    ocorrencias_por_dia: float
    por_dia: dict[str, int] = field(default_factory=dict)
    por_regime: dict[str, int] = field(default_factory=dict)
    por_campanha: dict[int, int] = field(default_factory=dict)


def estatisticas(database: Path, fault: str) -> EstatisticaHistorica:
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        total, primeira, ultima = conn.execute(
            "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM eventos WHERE fault = ?",
            (fault,),
        ).fetchone()

        if not total:
            return EstatisticaHistorica(fault, 0, None, None, 0, 0.0)

        por_dia = {
            linha["dia"]: linha["n"]
            for linha in conn.execute(
                "SELECT substr(created_at, 1, 10) AS dia, COUNT(*) AS n FROM eventos "
                "WHERE fault = ? GROUP BY dia ORDER BY dia",
                (fault,),
            )
        }
        por_regime = {
            linha["regime_rpm"]: linha["n"]
            for linha in conn.execute(
                "SELECT regime_rpm, COUNT(*) AS n FROM eventos WHERE fault = ? "
                "GROUP BY regime_rpm ORDER BY n DESC",
                (fault,),
            )
        }
        por_campanha = {
            linha["campanha"]: linha["n"]
            for linha in conn.execute(
                "SELECT campanha, COUNT(*) AS n FROM eventos WHERE fault = ? "
                "GROUP BY campanha ORDER BY campanha",
                (fault,),
            )
        }

    dias = len(por_dia)
    return EstatisticaHistorica(
        fault=fault,
        ocorrencias=total,
        primeira=primeira,
        ultima=ultima,
        dias_cobertos=dias,
        ocorrencias_por_dia=total / dias if dias else 0.0,
        por_dia=por_dia,
        por_regime=por_regime,
        por_campanha=por_campanha,
    )
