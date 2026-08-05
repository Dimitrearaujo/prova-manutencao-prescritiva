"""API HTTP da solucao.

Existe para que a solucao seja consumida pelo sistema que ja recebe os eventos
dos sensores, sem passar por tela. O app Streamlit e um cliente desta API em
processo, nao um caminho paralelo com regra propria.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from prescritiva.config import load_catalog, load_settings
from prescritiva.data.schema import VibrationEvent
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.integracao.repositorio import RepositorioDiagnosticos
from prescritiva.knowledge.extract import extract_all
from prescritiva.knowledge.store import BaseConhecimento

app = FastAPI(
    title="Manutencao Prescritiva",
    description="Diagnostico por similaridade historica com recuperacao de procedimento.",
    version="0.1.0",
)

LOGGER = logging.getLogger(__name__)

# O nome vem do cabecalho multipart, que e texto livre sob controle de quem
# chama. Allowlist em vez de lista de proibidos porque a variedade de formas de
# escrever um caminho (".." , "C:/", UNC, separador invertido, byte nulo) e
# grande demais para enumerar sem deixar buraco.
NOME_DOCUMENTO = re.compile(r"[A-Za-z0-9_.-]{1,80}\.pdf", re.IGNORECASE)

# Teto de tamanho para que um envio absurdo nao ocupe o disco da estacao nem
# entre na reindexacao. Nao impede o recebimento - isso e papel do servidor na
# frente - mas garante que nada alem deste tamanho chegue a docs_dir.
TAMANHO_MAXIMO_BYTES = 25 * 1024 * 1024
BLOCO_BYTES = 1024 * 1024

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


def _nome_seguro(filename: str | None, docs_dir: Path) -> str:
    """Aceita apenas um nome de arquivo simples, e recusa o resto.

    Sao tres conferencias porque cada uma cobre o ponto cego da anterior:

    1. Se o enviado difere do proprio nome-base, veio caminho junto. Recusar em
       vez de reduzir em silencio - quem mandou "../../evil.pdf" e recebeu 200
       fica acreditando que gravou onde pediu.
    2. A allowlist nao depende de interpretacao de caminho, que muda entre
       sistemas: no Linux "..\\..\\x.pdf" e um nome de arquivo valido e passaria
       inteiro pela conferencia 1.
    3. `is_relative_to` sobre o caminho resolvido e a confirmacao final, feita
       contra o sistema de arquivos de verdade e nao contra a string.
    """
    bruto = (filename or "").strip()
    nome = Path(bruto).name
    if bruto != nome:
        raise HTTPException(
            status_code=400, detail="Envie apenas o nome do arquivo, sem caminho."
        )
    if not NOME_DOCUMENTO.fullmatch(nome):
        raise HTTPException(
            status_code=400,
            detail=(
                "Nome de arquivo invalido: use letras, numeros, ponto, hifen ou "
                "sublinhado, terminando em .pdf."
            ),
        )
    if not (docs_dir / nome).resolve().is_relative_to(docs_dir.resolve()):
        raise HTTPException(status_code=400, detail="Nome de arquivo invalido.")
    return nome


def _gravar_em_blocos(arquivo: UploadFile, destino: Path) -> None:
    """Copia o envio para o disco em blocos, abortando ao passar do teto.

    Ler o arquivo inteiro em memoria antes de gravar deixa o tamanho do envio
    decidir quanta RAM o processo que atende a planta consome.
    """
    total = 0
    with destino.open("wb") as saida:
        while bloco := arquivo.file.read(BLOCO_BYTES):
            total += len(bloco)
            if total > TAMANHO_MAXIMO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Arquivo acima do limite de {TAMANHO_MAXIMO_BYTES // (1024 * 1024)} MB.",
                )
            saida.write(bloco)


def _exigir_pdf_legivel(caminho: Path) -> None:
    """Recusa com 400 o que o extrator nao conseguiria abrir depois.

    A validacao acontece enquanto o arquivo ainda esta na area temporaria. Se
    ela rodasse depois da gravacao em docs_dir, o arquivo ruim ficaria no
    diretorio e toda reindexacao seguinte morreria nele.
    """
    try:
        with fitz.open(caminho) as documento:
            paginas = documento.page_count
    except Exception as erro:  # noqa: BLE001 - qualquer falha de leitura vira 400, nunca 500
        LOGGER.warning("upload recusado, PDF ilegivel: %s", erro)
        raise HTTPException(
            status_code=400, detail="O arquivo enviado nao e um PDF legivel."
        ) from erro
    if paginas < 1:
        raise HTTPException(status_code=400, detail="O PDF enviado nao tem paginas.")


@app.post("/documentos")
def enviar_documento(
    arquivo: UploadFile = File(...), motor: MotorDiagnostico = Depends(obter_motor)
) -> dict[str, Any]:
    """Cadastra um procedimento novo e reindexa a base.

    E o outro lado da regra de cobertura: quando o sistema recusa prescrever por
    falta de documento, este endpoint e o caminho para resolver essa falta sem
    tocar em codigo.

    Sincrono de proposito. O corpo faz OCR (cerca de 25 s por pagina) e refaz o
    indice BM25; como `async def`, esse trabalho rodaria no event loop e deixaria
    /saude e /diagnosticar mudos por minutos. Sendo `def`, o FastAPI despacha ao
    threadpool e o resto da API continua respondendo durante o cadastro.
    """
    settings = load_settings()
    docs_dir = settings.paths.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    nome = _nome_seguro(arquivo.filename, docs_dir)

    # O arquivo so entra em docs_dir depois de validado: docs_dir e varrido por
    # glob a cada reindexacao, entao gravar antes de conferir e o que tornava um
    # unico envio ruim capaz de derrubar o cadastro em definitivo.
    with tempfile.TemporaryDirectory() as area:
        provisorio = Path(area) / nome
        _gravar_em_blocos(arquivo, provisorio)
        _exigir_pdf_legivel(provisorio)
        shutil.copyfile(provisorio, docs_dir / nome)

    # O cache de extracao e indexado por nome de arquivo. Sem invalidar, reenviar
    # o mesmo nome com o conteudo corrigido devolve o texto antigo: o cadastro
    # responde 200 e o sistema segue prescrevendo pela versao errada, que e o
    # oposto do que este endpoint existe para resolver.
    (settings.paths.knowledge_dir / f"{Path(nome).stem}.json").unlink(missing_ok=True)

    cfg = settings.knowledge
    extraidos = extract_all(
        docs_dir,
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
        "arquivo": nome,
        "documentos": len(base.documentos),
        "trechos": len(base.trechos),
        "cobertura": cobertura(motor),
    }
