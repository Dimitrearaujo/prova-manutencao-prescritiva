"""API HTTP da solucao.

Existe para que a solucao seja consumida pelo sistema que ja recebe os eventos
dos sensores, sem passar por tela. O app Streamlit e um cliente desta API em
processo, nao um caminho paralelo com regra propria.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from prescritiva.config import load_catalog, load_settings
from prescritiva.data.schema import VibrationEvent
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.knowledge.extract import extract_all
from prescritiva.knowledge.store import BaseConhecimento

app = FastAPI(
    title="Manutencao Prescritiva",
    description="Diagnostico por similaridade historica com recuperacao de procedimento.",
    version="0.1.0",
)

_motor: MotorDiagnostico | None = None


def obter_motor() -> MotorDiagnostico:
    global _motor
    if _motor is None:
        try:
            _motor = MotorDiagnostico.carregar()
        except FileNotFoundError as erro:
            raise HTTPException(
                status_code=503,
                detail="Indices ausentes. Execute scripts/ingest.py, build_index.py e build_knowledge.py.",
            ) from erro
    return _motor


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3)
    documento: str | None = None


@app.get("/saude")
def saude(motor: MotorDiagnostico = Depends(obter_motor)) -> dict[str, Any]:
    return {
        "status": "ok",
        "gerador": motor.gerador.nome,
        "documentos": len(motor.base.documentos),
        "trechos": len(motor.base.trechos),
        "regimes_indexados": sorted(motor.indice._metadados.keys()),
    }


@app.post("/diagnosticar")
def diagnosticar(evento: VibrationEvent, motor: MotorDiagnostico = Depends(obter_motor)) -> dict[str, Any]:
    return asdict(motor.diagnosticar(evento))


@app.post("/perguntar")
def perguntar(entrada: Pergunta, motor: MotorDiagnostico = Depends(obter_motor)) -> dict[str, Any]:
    return motor.perguntar(entrada.pergunta, documento=entrada.documento)


@app.get("/documentos")
def documentos(motor: MotorDiagnostico = Depends(obter_motor)) -> list[dict[str, Any]]:
    return [asdict(d) for d in motor.base.documentos]


@app.get("/cobertura")
def cobertura(motor: MotorDiagnostico = Depends(obter_motor)) -> list[dict[str, Any]]:
    saida = []
    for fault, entrada in load_catalog()["faults"].items():
        resultado = motor.base.cobertura(
            entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
        )
        saida.append({"fault": fault, "rotulo": entrada["rotulo"], **asdict(resultado)})
    return saida


@app.post("/documentos")
async def enviar_documento(
    arquivo: UploadFile = File(...), motor: MotorDiagnostico = Depends(obter_motor)
) -> dict[str, Any]:
    """Cadastra um procedimento novo e reindexa a base.

    E o outro lado da regra de cobertura: quando o sistema recusa prescrever por
    falta de documento, este endpoint e o caminho para resolver essa falta sem
    tocar em codigo.
    """
    if not (arquivo.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF.")

    settings = load_settings()
    destino = settings.paths.docs_dir / arquivo.filename
    destino.write_bytes(await arquivo.read())

    cfg = settings.knowledge
    extraidos = extract_all(
        settings.paths.docs_dir,
        settings.paths.knowledge_dir,
        dpi=cfg["ocr_dpi"],
        min_confidence=cfg["ocr_min_confidence"],
    )
    base = BaseConhecimento(
        tamanho_trecho=cfg["chunk_chars"],
        sobreposicao=cfg["chunk_overlap"],
        chars_escopo=cfg["scope_chars"],
        score_minimo_cobertura=cfg["coverage_min_score"],
    ).indexar(extraidos)
    base.salvar(settings.paths.index_dir)
    motor.base = base

    return {
        "arquivo": arquivo.filename,
        "documentos": len(base.documentos),
        "trechos": len(base.trechos),
        "cobertura": cobertura(motor),
    }
