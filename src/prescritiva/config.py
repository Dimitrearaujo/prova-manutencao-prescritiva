"""Carregamento de configuracao e resolucao de caminhos do projeto."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# O settings.yaml descreve a solucao; estas variaveis descrevem onde ela esta
# rodando. A separacao existe por causa do container: o Ollama fica fora dele,
# na estacao, porque o modelo precisa da GPU da maquina. Sem esta brecha o
# endereco do servico teria de ser editado no arquivo versionado, ou assado
# dentro da imagem, e o mesmo artefato deixaria de servir a duas maquinas.
_SOBRESCRITAS_ENV: dict[str, tuple[str, str]] = {
    "PRESCRITIVA_LLM_BASE_URL": ("llm", "base_url"),
    "PRESCRITIVA_LLM_MODEL": ("llm", "model"),
    # Mesmo motivo, do outro lado da fila: o broker MQTT e da planta, nao da
    # solucao. O endereco muda por instalacao e nao pode viver no arquivo
    # versionado nem dentro da imagem.
    "PRESCRITIVA_MQTT_HOST": ("mqtt", "host"),
}


@dataclass(frozen=True)
class Paths:
    raw_csv: Path
    docs_dir: Path
    processed_dir: Path
    index_dir: Path
    knowledge_dir: Path
    database: Path

    def ensure(self) -> None:
        for directory in (
            self.docs_dir,
            self.processed_dir,
            self.index_dir,
            self.knowledge_dir,
            self.database.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    paths: Paths
    ingest: dict[str, Any]
    similarity: dict[str, Any]
    knowledge: dict[str, Any]
    llm: dict[str, Any]
    # Integracao por fila e opcional: a solucao funciona por HTTP sem broker
    # nenhum. O default vazio deixa um settings.yaml sem o bloco continuar
    # valido, em vez de exigir configuracao de algo que talvez nao seja usado.
    mqtt: dict[str, Any] = field(default_factory=dict)


def _aplicar_sobrescritas_env(raw: dict[str, Any]) -> dict[str, Any]:
    for variavel, (secao, chave) in _SOBRESCRITAS_ENV.items():
        valor = os.environ.get(variavel, "").strip()
        # Variavel vazia e tratada como ausente: no compose uma chave sem valor
        # chega como string vazia, e apagar a URL do YAML seria pior do que
        # ignora-la.
        if valor:
            raw.setdefault(secao, {})[chave] = valor
    return raw


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    raw = _aplicar_sobrescritas_env(
        yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8"))
    )
    paths = Paths(**{k: PROJECT_ROOT / v for k, v in raw["paths"].items()})
    return Settings(
        paths=paths,
        ingest=raw["ingest"],
        similarity=raw["similarity"],
        knowledge=raw["knowledge"],
        llm=raw["llm"],
        mqtt=raw.get("mqtt", {}),
    )


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "fault_catalog.yaml").read_text(encoding="utf-8"))


def fault_label(fault: str) -> str:
    catalog = load_catalog()
    for section in ("faults", "estados_operacionais"):
        entry = catalog.get(section, {}).get(fault)
        if entry:
            return entry["rotulo"]
    return fault
