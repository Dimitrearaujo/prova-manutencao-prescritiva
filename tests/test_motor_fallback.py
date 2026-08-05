"""O gerador cair depois do boot nao pode custar o diagnostico.

O fallback para o gerador deterministico vivia so em llm/factory.py, avaliado
uma unica vez no carregamento - e o motor e singleton de processo. Se o Ollama
caisse com o servico ja no ar, a chamada crua em _defeito levantava
ConnectionError, /diagnosticar devolvia 500 e um diagnostico pronto (defeito
identificado, historico, cobertura, trechos recuperados) ia para o lixo. No
consumidor MQTT, pior: o evento ia para o topico de ERROS.
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.knowledge.store import BaseConhecimento
from prescritiva.similarity.index import IndiceSimilaridade


class GeradorQueCai:
    """O Ollama que respondeu ao probe no boot e morreu depois."""

    nome = "ollama:qwen2.5-3b"

    def disponivel(self) -> bool:
        return True

    def gerar(self, instrucao: str, pergunta: str) -> str:
        raise requests.ConnectionError("Connection refused - localhost:11434")


@pytest.fixture(scope="module")
def motor() -> MotorDiagnostico:
    settings = load_settings()
    if not (settings.paths.index_dir / "indice_similaridade.joblib").exists():
        pytest.skip("execute scripts/build_index.py e build_knowledge.py antes")
    return MotorDiagnostico(
        indice=IndiceSimilaridade.carregar(settings.paths.index_dir),
        base=BaseConhecimento.carregar(settings.paths.index_dir),
        database=settings.paths.database,
        gerador=GeradorQueCai(),
        top_k=settings.knowledge["retrieval_top_k"],
    )


# Evento fixo, e nao sorteado: e um rolamento de anel interno que atravessa os
# tres portoes e chega a defeito_documentado com Doc1 e 96% de consenso. Sortear
# aqui faria o teste do fallback depender de o sorteio nao cair na regra de
# rejeicao, que e o defeito que a auditoria encontrou em tests/test_engine.py.
EVENTO_DOCUMENTADO = 130348


@pytest.fixture(scope="module")
def evento_com_procedimento() -> dict:
    caminho = load_settings().paths.processed_dir / "eventos.parquet"
    if not caminho.exists():
        pytest.skip("execute scripts/ingest.py antes")
    eventos = pd.read_parquet(caminho)
    linha = eventos[eventos["id"] == EVENTO_DOCUMENTADO].iloc[0]
    return {k: v for k, v in linha.to_dict().items() if k not in {"fault", "fault_original"}}


def test_diagnostico_sai_mesmo_com_o_gerador_fora_do_ar(motor, evento_com_procedimento):
    diagnostico = motor.diagnosticar(evento_com_procedimento)

    assert diagnostico.situacao == "defeito_documentado"
    assert diagnostico.cobertura["documento"] == "Doc1"
    assert diagnostico.trechos
    # O texto sai recortado do procedimento, e a troca aparece na resposta em vez
    # de acontecer em silencio.
    assert diagnostico.instrucoes
    assert "fallback" in diagnostico.gerador
    assert "deterministico" in diagnostico.gerador


def test_chat_sai_mesmo_com_o_gerador_fora_do_ar(motor):
    resultado = motor.perguntar("como corrigir desalinhamento de eixo?")

    assert resultado["situacao"] == "respondido"
    assert resultado["trechos"]
    assert "fallback" in resultado["gerador"]
    assert resultado["resposta"]
