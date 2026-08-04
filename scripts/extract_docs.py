"""Etapa 1 do conhecimento: PDF -> texto (OCR quando necessario), com cache."""

from __future__ import annotations

import logging

from prescritiva.config import load_settings
from prescritiva.knowledge.extract import extract_all


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = load_settings()
    settings.paths.ensure()
    documentos = extract_all(
        settings.paths.docs_dir,
        settings.paths.knowledge_dir,
        dpi=settings.knowledge["ocr_dpi"],
        min_confidence=settings.knowledge["ocr_min_confidence"],
    )
    print(f"\n{'documento':<12} {'paginas':>8} {'chars':>8}  metodo")
    for doc in documentos:
        print(f"{doc.nome:<12} {doc.paginas:>8} {len(doc.texto):>8}  {doc.metodo}")
        print(f"             titulo: {doc.titulo}")


if __name__ == "__main__":
    main()
