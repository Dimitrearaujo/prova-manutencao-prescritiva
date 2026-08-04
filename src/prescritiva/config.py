"""Carregamento de configuracao e resolucao de caminhos do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


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


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    raw = yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8"))
    paths = Paths(**{k: PROJECT_ROOT / v for k, v in raw["paths"].items()})
    return Settings(
        paths=paths,
        ingest=raw["ingest"],
        similarity=raw["similarity"],
        knowledge=raw["knowledge"],
        llm=raw["llm"],
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
