"""API HTTP da solucao.

Existe para que a solucao seja consumida pelo sistema que ja recebe os eventos
dos sensores, sem passar por tela. O app Streamlit e um cliente desta API em
processo, nao um caminho paralelo com regra propria.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from prescritiva.config import load_catalog, load_settings
from prescritiva.data.schema import VibrationEvent
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.integracao.repositorio import RepositorioDiagnosticos
from prescritiva.knowledge.cadastro import CadastroInvalido, cadastrar_documento

app = FastAPI(
    title="Manutencao Prescritiva",
    description="Diagnostico por similaridade historica com recuperacao de procedimento.",
    version="0.1.0",
)

LOGGER = logging.getLogger(__name__)

_motor: MotorDiagnostico | None = None
_repositorio: RepositorioDiagnosticos | None = None


def obter_repositorio() -> RepositorioDiagnosticos:
    global _repositorio
    if _repositorio is None:
        _repositorio = RepositorioDiagnosticos(load_settings().paths.database)
    return _repositorio


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
def diagnosticar(
    evento: VibrationEvent,
    motor: MotorDiagnostico = Depends(obter_motor),
    repositorio: RepositorioDiagnosticos = Depends(obter_repositorio),
) -> dict[str, Any]:
    """Diagnostica e registra.

    O registro acontece aqui e nao dentro do motor porque decidir o defeito nao
    pode depender de disco disponivel. Se a gravacao falha, a resposta sai
    mesmo assim: quem perguntou precisa dela agora.
    """
    diagnostico = motor.diagnosticar(evento)
    resposta = asdict(diagnostico)
    try:
        resposta["registro_id"] = repositorio.registrar(
            diagnostico, evento_id=evento.id, origem="api"
        )
    except Exception:  # noqa: BLE001
        LOGGER.exception("falha ao gravar o diagnostico do evento %s", evento.id)
        resposta["registro_id"] = None
    return resposta


@app.get("/diagnosticos")
def diagnosticos(
    limite: int = 50,
    situacao: str | None = None,
    fault: str | None = None,
    repositorio: RepositorioDiagnosticos = Depends(obter_repositorio),
) -> dict[str, Any]:
    """Historico de diagnosticos emitidos, seja qual for a porta de entrada.

    Fecha o ciclo do enunciado: a solucao consulta o banco da empresa e devolve
    para o banco da empresa. A contagem por situacao e o indicador de deriva -
    padrao_desconhecido subindo significa que a maquina saiu da condicao em que
    o indice foi construido.
    """
    return {
        "por_situacao": repositorio.contar_por_situacao(),
        "registros": [asdict(r) for r in repositorio.listar(
            limite=limite, situacao=situacao, fault=fault
        )],
    }


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
def enviar_documento(
    arquivo: UploadFile = File(...),
    motor: MotorDiagnostico = Depends(obter_motor),
    x_prescritiva_key: str | None = Header(default=None),
) -> dict[str, Any]:
    """Cadastra um procedimento novo e reindexa a base.

    E o outro lado da regra de cobertura: quando o sistema recusa prescrever por
    falta de documento, este endpoint e o caminho para resolver essa falta sem
    tocar em codigo. A validacao, a gravacao e a reindexacao moram em
    `prescritiva.knowledge.cadastro`, compartilhadas com o painel Streamlit -
    aqui so cabe traduzir `CadastroInvalido` para HTTP.

    `X-Prescritiva-Key` so e exigido se `cadastro.chave_acesso` estiver
    configurada (`prescritiva.knowledge.cadastro._exigir_chave_valida` decide):
    minha arquitetura promete rodar numa planta segmentada, entao reescrever o
    procedimento que o tecnico vai seguir nao pode ser acao de quem quer que
    alcance esta porta.

    Sincrono de proposito. O corpo faz OCR (cerca de 25 s por pagina) e refaz o
    indice BM25; como `async def`, esse trabalho rodaria no event loop e deixaria
    /saude e /diagnosticar mudos por minutos. Sendo `def`, o FastAPI despacha ao
    threadpool e o resto da API continua respondendo durante o cadastro.
    """
    settings = load_settings()
    try:
        resultado = cadastrar_documento(
            arquivo.file, arquivo.filename, settings, chave_apresentada=x_prescritiva_key
        )
    except CadastroInvalido as erro:
        raise HTTPException(status_code=erro.status_code, detail=erro.mensagem) from erro
    motor.base = resultado.base

    return {
        "arquivo": resultado.nome,
        "documentos": len(resultado.base.documentos),
        "trechos": len(resultado.base.trechos),
        "cobertura": cobertura(motor),
    }
