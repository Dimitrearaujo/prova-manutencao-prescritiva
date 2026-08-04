from __future__ import annotations

from prescritiva.llm.base import GeradorTexto
from prescritiva.llm.deterministico import GeradorDeterministico
from prescritiva.llm.ollama_client import OllamaGerador


def construir_gerador(cfg: dict) -> GeradorTexto:
    """Devolve o gerador configurado, ou o deterministico se ele nao responder."""
    if cfg.get("provider") == "ollama":
        gerador = OllamaGerador(
            modelo=cfg["model"],
            base_url=cfg.get("base_url", "http://localhost:11434"),
            temperatura=cfg.get("temperature", 0.1),
            timeout_s=cfg.get("timeout_s", 300),
            max_tokens=cfg.get("max_tokens", 600),
            contexto=cfg.get("num_ctx", 4096),
        )
        if gerador.disponivel():
            return gerador
    return GeradorDeterministico()
