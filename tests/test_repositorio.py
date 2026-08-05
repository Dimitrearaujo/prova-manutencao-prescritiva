"""Testes do registro de diagnosticos, contra um SQLite temporario.

Nao dependem dos indices nem do modelo de linguagem: o repositorio recebe um
diagnostico pronto e a unica pergunta e se o que sai do banco e igual ao que
entrou.
"""

from __future__ import annotations

import sqlite3

import pytest

from prescritiva.diagnosis.engine import Diagnostico
from prescritiva.integracao.repositorio import RepositorioDiagnosticos


def _diagnostico(
    situacao: str = "defeito_documentado",
    *,
    distancia_media: float = 1.23,
    documento: str | None = "Doc2",
    instrucoes: str = "1. Bloquear o equipamento.",
) -> Diagnostico:
    return Diagnostico(
        situacao=situacao,
        fault="desalinhado",
        rotulo="Desalinhamento de eixo",
        confianca=0.87,
        regime_rpm="1000rpm",
        e_defeito=True,
        mensagem="mensagem de teste",
        similaridade={"distancia_media": distancia_media},
        cobertura={"documento": documento},
        instrucoes=instrucoes,
    )


@pytest.fixture
def repositorio(tmp_path) -> RepositorioDiagnosticos:
    return RepositorioDiagnosticos(tmp_path / "prescritiva.db")


def test_grava_e_le_de_volta_os_mesmos_valores(repositorio):
    id_registro = repositorio.registrar(_diagnostico(), evento_id=4242, origem="mqtt")

    registro = repositorio.obter(id_registro)
    assert registro is not None
    assert registro.evento_id == 4242
    assert registro.origem == "mqtt"
    assert registro.situacao == "defeito_documentado"
    assert registro.fault == "desalinhado"
    assert registro.confianca == pytest.approx(0.87)
    assert registro.distancia_media == pytest.approx(1.23)
    assert registro.documento == "Doc2"
    assert registro.instrucoes.startswith("1. Bloquear")
    assert registro.registrado_em


def test_evento_sem_id_e_aceito(repositorio):
    """Evento novo de verdade nao tem id: chega do sensor, nao do historico."""
    id_registro = repositorio.registrar(_diagnostico(), origem="api")

    registro = repositorio.obter(id_registro)
    assert registro is not None
    assert registro.evento_id is None


def test_distancia_infinita_vira_nulo(repositorio):
    """Sem historico no regime a distancia e infinita.

    Guardar infinito envenenaria qualquer media e ordenacao feita depois sobre a
    tabela, alem de quebrar a exportacao JSON de quem consome o banco.
    """
    id_registro = repositorio.registrar(
        _diagnostico("padrao_desconhecido", distancia_media=float("inf"), documento=None)
    )

    registro = repositorio.obter(id_registro)
    assert registro is not None
    assert registro.distancia_media is None
    assert registro.documento is None


def test_os_quatro_desfechos_sao_registrados(repositorio):
    """Nenhum desfecho e descartado, inclusive os dois em que nada e prescrito."""
    for situacao in (
        "defeito_documentado",
        "defeito_sem_documentacao",
        "estado_operacional",
        "padrao_desconhecido",
    ):
        repositorio.registrar(_diagnostico(situacao))

    assert repositorio.contar_por_situacao() == {
        "defeito_documentado": 1,
        "defeito_sem_documentacao": 1,
        "estado_operacional": 1,
        "padrao_desconhecido": 1,
    }


def test_o_mesmo_evento_rediagnosticado_gera_dois_registros(repositorio):
    """A tabela e trilha de auditoria, nao estado atual.

    Depois de cadastrar um procedimento, o mesmo evento passa de recusado a
    prescrito. As duas respostas precisam sobreviver: sobrescrever apagaria a
    justificativa da decisao tomada na primeira vez.
    """
    repositorio.registrar(_diagnostico("defeito_sem_documentacao"), evento_id=99)
    repositorio.registrar(_diagnostico("defeito_documentado"), evento_id=99)

    registros = repositorio.listar(evento_id=99)
    assert len(registros) == 2
    assert registros[0].situacao == "defeito_documentado", "o mais recente vem primeiro"


def test_listar_filtra_e_limita(repositorio):
    for _ in range(3):
        repositorio.registrar(_diagnostico("estado_operacional"))
    repositorio.registrar(_diagnostico("defeito_documentado"))

    assert len(repositorio.listar(situacao="estado_operacional")) == 3
    assert len(repositorio.listar(limite=2)) == 2
    assert repositorio.listar(fault="inexistente") == []


def test_criar_esquema_e_idempotente(tmp_path):
    """Subir o consumidor duas vezes no mesmo banco nao pode falhar nem apagar."""
    caminho = tmp_path / "prescritiva.db"
    RepositorioDiagnosticos(caminho).registrar(_diagnostico())

    segundo = RepositorioDiagnosticos(caminho)
    assert len(segundo.listar()) == 1


def test_convive_com_a_tabela_de_eventos(tmp_path):
    """O registro entra na base da empresa, nao numa base paralela."""
    caminho = tmp_path / "prescritiva.db"
    with sqlite3.connect(caminho) as conn:
        conn.execute("CREATE TABLE eventos (id INTEGER, fault TEXT)")
        conn.execute("INSERT INTO eventos VALUES (1, 'desalinhado')")

    RepositorioDiagnosticos(caminho).registrar(_diagnostico(), evento_id=1)

    with sqlite3.connect(caminho) as conn:
        tabelas = {nome for (nome,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        juncao = conn.execute(
            "SELECT e.fault, d.situacao FROM eventos e JOIN diagnosticos d ON d.evento_id = e.id"
        ).fetchall()
    assert {"eventos", "diagnosticos"} <= tabelas
    assert juncao == [("desalinhado", "defeito_documentado")]
