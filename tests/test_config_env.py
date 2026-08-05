"""Garante que a configuracao de ambiente sobrescreve a do arquivo.

Este e o unico ponto de contato entre a solucao e a infraestrutura que a
hospeda: dentro do container o Ollama nao esta em localhost, e sem esta
sobrescrita a imagem so serviria a maquina cujo endereco estivesse escrito no
settings.yaml. Um teste aqui evita que a regressao apareca so no deploy.
"""

from __future__ import annotations

import pytest

from prescritiva.config import _aplicar_sobrescritas_env, load_settings


@pytest.fixture
def bruto() -> dict:
    return {"llm": {"base_url": "http://localhost:11434", "model": "qwen2.5:3b"}}


def test_sem_variavel_o_arquivo_manda(bruto, monkeypatch):
    monkeypatch.delenv("PRESCRITIVA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PRESCRITIVA_LLM_MODEL", raising=False)

    assert _aplicar_sobrescritas_env(bruto)["llm"]["base_url"] == "http://localhost:11434"


def test_variavel_aponta_para_o_ollama_do_host(bruto, monkeypatch):
    monkeypatch.setenv("PRESCRITIVA_LLM_BASE_URL", "http://host.docker.internal:11434")

    assert _aplicar_sobrescritas_env(bruto)["llm"]["base_url"] == "http://host.docker.internal:11434"


def test_variavel_vazia_nao_apaga_o_valor_do_arquivo(bruto, monkeypatch):
    """No compose, uma chave declarada sem valor chega como string vazia."""
    monkeypatch.setenv("PRESCRITIVA_LLM_BASE_URL", "")

    assert _aplicar_sobrescritas_env(bruto)["llm"]["base_url"] == "http://localhost:11434"


def test_modelo_tambem_e_sobrescrivel(bruto, monkeypatch):
    """A estacao industrial com GPU de 16 GB comporta modelo maior que a de demo."""
    monkeypatch.setenv("PRESCRITIVA_LLM_MODEL", "qwen2.5:14b")

    assert _aplicar_sobrescritas_env(bruto)["llm"]["model"] == "qwen2.5:14b"


def test_settings_reais_leem_a_variavel(monkeypatch):
    """Fecha o circuito: a sobrescrita chega ao objeto que o motor consome."""
    monkeypatch.setenv("PRESCRITIVA_LLM_BASE_URL", "http://host.docker.internal:11434")
    load_settings.cache_clear()
    try:
        assert load_settings().llm["base_url"] == "http://host.docker.internal:11434"
    finally:
        # O cache e global: deixa-lo sujo contaminaria os testes seguintes.
        load_settings.cache_clear()


def test_chave_de_cadastro_e_sobrescrivel(monkeypatch):
    """A chave e segredo da instalacao: nao pode viver so no settings.yaml versionado."""
    bruto = {"cadastro": {"chave_acesso": None}}
    monkeypatch.setenv("PRESCRITIVA_CADASTRO_KEY", "planta-nova-2026")

    assert _aplicar_sobrescritas_env(bruto)["cadastro"]["chave_acesso"] == "planta-nova-2026"


def test_settings_reais_leem_a_chave_de_cadastro(monkeypatch):
    monkeypatch.setenv("PRESCRITIVA_CADASTRO_KEY", "planta-nova-2026")
    load_settings.cache_clear()
    try:
        assert load_settings().cadastro["chave_acesso"] == "planta-nova-2026"
    finally:
        load_settings.cache_clear()
