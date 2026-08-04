"""Extracao de texto dos documentos orientativos.

Os PDFs fornecidos nao sao homogeneos: cinco tem camada de texto nativa e um
(Doc1, rolamentos) e um PDF de paginas rasterizadas, sem texto algum. A
extracao decide por documento qual caminho seguir, e o resultado do OCR fica
em cache porque custa cerca de 25 s por pagina.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

from prescritiva.text import collapse_whitespace

logger = logging.getLogger(__name__)

# Abaixo disto a camada de texto e considerada inexistente (paginas escaneadas
# costumam devolver apenas espacos e quebras de linha).
_MIN_CHARS_PER_PAGE = 40


@dataclass
class ExtractedDocument:
    nome: str
    arquivo: str
    paginas: int
    metodo: str
    texto: str

    @property
    def titulo(self) -> str:
        """O titulo dos procedimentos quebra em varias linhas antes da secao 1."""
        partes: list[str] = []
        for linha in self.texto.splitlines():
            limpa = linha.strip()
            if not limpa:
                continue
            if re.match(r"^\d+(\.\d+)*\.?\s", limpa):
                break
            partes.append(limpa)
            if len(" ".join(partes)) > 90:
                break
        return " ".join(partes) or self.nome


def _native_text(doc: fitz.Document) -> str:
    return "\n".join(page.get_text() for page in doc)


def _ocr_text(pdf_path: Path, dpi: int, min_confidence: float) -> str:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    doc = fitz.open(pdf_path)
    paginas: list[str] = []
    for numero, page in enumerate(doc, start=1):
        image = page.get_pixmap(dpi=dpi).tobytes("png")
        result, _ = engine(image)
        linhas = [
            texto.strip()
            for _, texto, score in (result or [])
            if score >= min_confidence and texto.strip()
        ]
        logger.info("OCR %s pagina %d/%d: %d linhas", pdf_path.name, numero, len(doc), len(linhas))
        paginas.append("\n".join(linhas))
    return "\n\n".join(paginas)


def extract_document(
    pdf_path: Path,
    cache_dir: Path,
    *,
    dpi: int = 300,
    min_confidence: float = 0.5,
    force: bool = False,
) -> ExtractedDocument:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{pdf_path.stem}.json"
    if cache_file.exists() and not force:
        return ExtractedDocument(**json.loads(cache_file.read_text(encoding="utf-8")))

    doc = fitz.open(pdf_path)
    texto = _native_text(doc)
    metodo = "texto-nativo"
    if len(texto.strip()) < _MIN_CHARS_PER_PAGE * len(doc):
        logger.info("%s sem camada de texto util, aplicando OCR", pdf_path.name)
        texto = _ocr_text(pdf_path, dpi, min_confidence)
        metodo = f"ocr-rapidocr@{dpi}dpi"

    extraido = ExtractedDocument(
        nome=pdf_path.stem,
        arquivo=pdf_path.name,
        paginas=len(doc),
        metodo=metodo,
        texto=collapse_whitespace(texto),
    )
    cache_file.write_text(
        json.dumps(asdict(extraido), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return extraido


def extract_all(
    docs_dir: Path, cache_dir: Path, *, dpi: int = 300, min_confidence: float = 0.5
) -> list[ExtractedDocument]:
    return [
        extract_document(pdf, cache_dir, dpi=dpi, min_confidence=min_confidence)
        for pdf in sorted(docs_dir.glob("*.pdf"))
    ]
