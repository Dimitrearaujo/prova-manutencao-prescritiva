"""Persistencia dos diagnosticos emitidos.

O historico de eventos entra pelo SQLite, mas ate aqui o diagnostico saia apenas
pela resposta HTTP ou pela tela: quem perguntou viu a resposta, e mais ninguem.
Numa planta isso nao fecha o ciclo. O diagnostico e o registro que a manutencao
consulta depois para justificar uma parada, medir reincidencia e alimentar o
CMMS, e por isso ele volta para a mesma base de onde o historico e lido, em
tabela propria.

Fica fora do motor de proposito. Decidir o defeito nao pode depender de disco
disponivel: quem chama o motor escolhe se registra, e uma falha de gravacao nao
pode reter a resposta que o tecnico esta esperando.
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    # Import so para tipo: gravar nao exige o motor carregado, e manter essa
    # fronteira permite usar o repositorio num processo que so le o banco, sem
    # arrastar sklearn e os indices para dentro dele.
    from prescritiva.diagnosis.engine import Diagnostico

# Sem chave unica por evento, de proposito. A tabela e um log append-only: o
# mesmo evento pode ser rediagnosticado depois que um procedimento novo e
# cadastrado, e a resposta muda. Sobrescrever apagaria a trilha de auditoria que
# justifica a decisao tomada naquele momento. Quem quer "o diagnostico atual"
# pega o registro mais recente do evento.
ESQUEMA = """
CREATE TABLE IF NOT EXISTS diagnosticos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    evento_id       INTEGER,
    registrado_em   TEXT    NOT NULL,
    origem          TEXT    NOT NULL,
    situacao        TEXT    NOT NULL,
    fault           TEXT,
    rotulo          TEXT,
    confianca       REAL,
    distancia_media REAL,
    regime_rpm      TEXT,
    documento       TEXT,
    instrucoes      TEXT    NOT NULL DEFAULT ''
)
"""

INDICES = (
    "CREATE INDEX IF NOT EXISTS ix_diagnosticos_situacao ON diagnosticos(situacao)",
    "CREATE INDEX IF NOT EXISTS ix_diagnosticos_fault ON diagnosticos(fault)",
    "CREATE INDEX IF NOT EXISTS ix_diagnosticos_evento ON diagnosticos(evento_id)",
    "CREATE INDEX IF NOT EXISTS ix_diagnosticos_registrado ON diagnosticos(registrado_em)",
)

_CAMPOS = (
    "id",
    "evento_id",
    "registrado_em",
    "origem",
    "situacao",
    "fault",
    "rotulo",
    "confianca",
    "distancia_media",
    "regime_rpm",
    "documento",
    "instrucoes",
)


@dataclass(frozen=True)
class RegistroDiagnostico:
    """Uma linha da tabela, do jeito que sai na leitura."""

    id: int
    evento_id: int | None
    registrado_em: str
    origem: str
    situacao: str
    fault: str | None
    rotulo: str | None
    confianca: float | None
    distancia_media: float | None
    regime_rpm: str | None
    documento: str | None
    instrucoes: str


class RepositorioDiagnosticos:
    def __init__(self, database: Path | str) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.criar_esquema()

    def criar_esquema(self) -> None:
        """Cria tabela e indices se faltarem.

        Roda na construcao para que subir o consumidor num banco recem-ingerido
        funcione sem passo manual de migracao. E idempotente, entao chamar de
        novo nao custa nada.
        """
        with closing(self._conectar()) as conn, conn:
            conn.execute(ESQUEMA)
            for indice in INDICES:
                conn.execute(indice)

    def registrar(
        self,
        diagnostico: "Diagnostico",
        *,
        evento_id: int | None = None,
        origem: str = "api",
        registrado_em: str | None = None,
    ) -> int:
        """Grava um diagnostico e devolve o id do registro.

        Guarda o que sustenta a decisao (situacao, defeito, confianca, distancia
        media e documento usado) e a prescricao entregue, nao o diagnostico
        inteiro. Vizinhos e trechos ficam de fora porque sao reproduziveis a
        partir do evento e do indice versionado, e inflariam a tabela em ordens
        de grandeza sem acrescentar informacao nova.
        """
        linha = (
            evento_id,
            registrado_em or agora(),
            origem,
            diagnostico.situacao,
            diagnostico.fault,
            diagnostico.rotulo,
            _real_ou_nulo(diagnostico.confianca),
            _real_ou_nulo(diagnostico.similaridade.get("distancia_media")),
            diagnostico.regime_rpm,
            diagnostico.cobertura.get("documento"),
            diagnostico.instrucoes or "",
        )
        with closing(self._conectar()) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO diagnosticos (evento_id, registrado_em, origem, situacao, fault, "
                "rotulo, confianca, distancia_media, regime_rpm, documento, instrucoes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                linha,
            )
            return int(cursor.lastrowid or 0)

    def obter(self, id_registro: int) -> RegistroDiagnostico | None:
        with closing(self._conectar()) as conn, conn:
            linha = conn.execute(
                f"SELECT {', '.join(_CAMPOS)} FROM diagnosticos WHERE id = ?", (id_registro,)
            ).fetchone()
        return RegistroDiagnostico(*linha) if linha else None

    def listar(
        self,
        *,
        limite: int = 50,
        situacao: str | None = None,
        fault: str | None = None,
        evento_id: int | None = None,
    ) -> list[RegistroDiagnostico]:
        """Os registros mais recentes primeiro, opcionalmente filtrados."""
        filtros: list[str] = []
        parametros: list[Any] = []
        for coluna, valor in (("situacao", situacao), ("fault", fault), ("evento_id", evento_id)):
            if valor is not None:
                filtros.append(f"{coluna} = ?")
                parametros.append(valor)

        onde = f" WHERE {' AND '.join(filtros)}" if filtros else ""
        parametros.append(limite)
        with closing(self._conectar()) as conn, conn:
            linhas = conn.execute(
                f"SELECT {', '.join(_CAMPOS)} FROM diagnosticos{onde} ORDER BY id DESC LIMIT ?",
                parametros,
            ).fetchall()
        return [RegistroDiagnostico(*linha) for linha in linhas]

    def contar_por_situacao(self) -> dict[str, int]:
        """Quanto de cada desfecho ja foi emitido.

        E a medida operacional que o enunciado pede de forma indireta: uma
        fatia crescente de padrao_desconhecido significa que a maquina saiu da
        condicao em que o indice foi construido, e uma fatia grande de
        defeito_sem_documentacao aponta exatamente quais procedimentos faltam
        cadastrar.
        """
        with closing(self._conectar()) as conn, conn:
            return {
                situacao: total
                for situacao, total in conn.execute(
                    "SELECT situacao, COUNT(*) FROM diagnosticos GROUP BY situacao "
                    "ORDER BY COUNT(*) DESC"
                )
            }

    def _conectar(self) -> sqlite3.Connection:
        """Conexao de vida curta, uma por operacao.

        Chamada sempre como `with closing(self._conectar()) as conn, conn:`. O
        `closing` fecha e o segundo gerenciador confirma a transacao: usar so
        `with sqlite3.connect(...)` confirmaria sem fechar, e um servico que
        fica meses no ar gravando um diagnostico por evento vazaria descritor
        ate cair.

        Nao existe conexao compartilhada porque o consumidor grava na thread do
        trabalhador e a API na dela, e conexao do sqlite3 nao atravessa thread.
        """
        return sqlite3.connect(self.database)


def agora() -> str:
    """Instante do registro no mesmo formato usado pela coluna created_at.

    A tabela eventos guarda "2023-03-15 10:00:00+00:00", entao os recortes por
    dia (`substr(...,1,10)`) e as comparacoes entre as duas tabelas funcionam
    sem conversao. UTC porque planta com turno atravessando meia-noite e
    horario de verao tornam hora local uma fonte de erro silencioso.
    """
    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def _real_ou_nulo(valor: Any) -> float | None:
    """Descarta o que nao e numero finito.

    `distancia_media` vale infinito quando nao ha historico no regime do evento.
    Infinito guardado num banco relacional envenena media e ordenacao, e quebra
    a exportacao JSON de qualquer consumidor.
    """
    if valor is None:
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) else None
