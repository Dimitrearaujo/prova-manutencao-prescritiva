"""Testes da API HTTP, contra os artefatos construidos.

A API e a porta que o sistema da empresa usa: o Streamlit e um cliente dela, e
o consumidor MQTT chama o mesmo motor. Ate aqui nenhum teste passava por ela, e
os dois defeitos mais caros do endpoint de upload (travessia de caminho e um PDF
ruim derrubando o cadastro em definitivo) sobreviveram justamente por isso.

O gerador de texto e trocado pelo deterministico em todos os testes: o desfecho
que interessa e o do motor - reconhecer, cobrir e recusar -, nao a redacao do
modelo de linguagem, e depender do Ollama tornaria a suite nao reproduzivel.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prescritiva.api import main as api
from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.integracao.repositorio import RepositorioDiagnosticos
from prescritiva.knowledge import cadastro as cadastro_mod
from prescritiva.knowledge.extract import extract_all
from prescritiva.llm.deterministico import GeradorDeterministico

# Quantos eventos do mesmo rotulo sao consultados ate achar um que chegue ao
# desfecho procurado. O motor rejeita parte dos eventos por consenso, entao
# fixar um id unico deixaria o teste refem do sorteio - foi assim que o unico
# teste do desfecho principal acabou sendo pulado em silencio.
LIMITE_BUSCA = 40

EVENTO_ABSURDO: dict[str, float] = {
    "rpm": 1000.0,
    "z_rms_velocity_mm_s": 900.0,
    "x_rms_velocity_mm_s": 850.0,
    "temperature_c": 240.0,
    "z_peak_acceleration_g": 400.0,
    "x_peak_acceleration_g": 380.0,
    "z_peak_vel_comp_freq_hz": 61.0,
    "x_peak_vel_comp_freq_hz": 61.0,
    "z_rms_acceleration_g": 300.0,
    "x_rms_acceleration_g": 290.0,
    "z_kurtosis": 90.0,
    "x_kurtosis": 88.0,
    "z_crest_factor": 70.0,
    "x_crest_factor": 68.0,
    "z_peak_velocity_mm_s": 950.0,
    "x_peak_velocity_mm_s": 940.0,
    "z_high_freq_rms_accel_g": 200.0,
    "x_high_freq_rms_accel_g": 210.0,
}

NOMES_MALICIOSOS = [
    "../../evil.pdf",
    "..\\..\\evil.pdf",
    "C:/Windows/Temp/evil.pdf",
    "/etc/cron.d/evil.pdf",
    "../config/settings.pdf",
    "evil.pdf\x00.txt",
    "sem_extensao",
    "script.pdf.exe",
]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def motor() -> MotorDiagnostico:
    settings = load_settings()
    if not (settings.paths.index_dir / "indice_similaridade.joblib").exists():
        pytest.skip("execute scripts/build_index.py e build_knowledge.py antes")
    carregado = MotorDiagnostico.carregar()
    carregado.gerador = GeradorDeterministico()
    return carregado


@pytest.fixture(scope="module")
def eventos() -> pd.DataFrame:
    caminho = load_settings().paths.processed_dir / "eventos.parquet"
    if not caminho.exists():
        pytest.skip("execute scripts/ingest.py antes")
    return pd.read_parquet(caminho)


@pytest.fixture
def cliente(motor: MotorDiagnostico, tmp_path: Path):
    """Cliente com o motor real e um banco de diagnosticos descartavel.

    O repositorio aponta para tmp_path para que a contagem de /diagnosticos seja
    a deste teste e nao a do banco de desenvolvimento.
    """
    repositorio = RepositorioDiagnosticos(tmp_path / "diagnosticos.db")
    api.app.dependency_overrides[api.obter_motor] = lambda: motor
    api.app.dependency_overrides[api.obter_repositorio] = lambda: repositorio
    try:
        with TestClient(api.app) as testclient:
            yield testclient
    finally:
        api.app.dependency_overrides.clear()


def _payload(eventos: pd.DataFrame, linha: pd.Series) -> dict[str, Any]:
    dados = linha.to_dict()
    for descartar in ("fault", "fault_original"):
        dados.pop(descartar, None)
    saida: dict[str, Any] = {}
    for chave, valor in dados.items():
        if isinstance(valor, pd.Timestamp):
            saida[chave] = valor.isoformat()
        elif hasattr(valor, "item"):
            saida[chave] = valor.item()
        else:
            saida[chave] = valor
    return saida


def _evento_com_desfecho(
    motor: MotorDiagnostico, eventos: pd.DataFrame, fault: str, situacao: str
) -> dict[str, Any]:
    """Primeiro evento do rotulo que chega ao desfecho pedido.

    Falha o teste se nenhum chegar, em vez de pular: se o rotulo deixar de
    alcancar o desfecho, isso e a regressao que o teste existe para pegar.
    """
    candidatos = eventos[eventos["fault"] == fault].sort_values("id").head(LIMITE_BUSCA)
    for _, linha in candidatos.iterrows():
        payload = _payload(eventos, linha)
        if motor.diagnosticar(payload).situacao == situacao:
            return payload
    pytest.fail(f"nenhum dos {LIMITE_BUSCA} eventos de {fault} chegou a {situacao}")


# --------------------------------------------------------------------------
# /saude e os quatro desfechos de /diagnosticar
# --------------------------------------------------------------------------


def test_saude_relata_o_estado_do_motor(cliente):
    corpo = cliente.get("/saude").json()
    assert corpo["status"] == "ok"
    # >= e nao ==: os seis procedimentos entregues tem de estar na base, mas
    # cadastrar um setimo pelo endpoint e uso normal do sistema e nao pode
    # quebrar o teste. Perder documento continua sendo pego.
    assert corpo["documentos"] >= 6
    assert corpo["trechos"] > 0
    assert corpo["regimes_indexados"]


def test_defeito_com_procedimento_prescreve(cliente, motor, eventos):
    """Desfecho 1: reconhece, acha o procedimento e entrega a instrucao."""
    payload = _evento_com_desfecho(motor, eventos, "desalinhado", "defeito_documentado")
    corpo = cliente.post("/diagnosticar", json=payload).json()

    assert corpo["situacao"] == "defeito_documentado"
    assert corpo["fault"] == "desalinhado"
    assert corpo["cobertura"]["documento"] == "Doc2"
    assert corpo["instrucoes"]
    assert corpo["trechos"]
    # A prescricao so pode citar o documento que cobre o defeito.
    assert {t["documento"] for t in corpo["trechos"]} == {"Doc2"}


@pytest.mark.parametrize("fault", ["eccentric_rotor", "ventoinha"])
def test_defeito_sem_procedimento_recusa_e_pede_cadastro(cliente, motor, eventos, fault):
    """Desfecho 2: a regra dura do enunciado, pela porta HTTP.

    Sem `if` de guarda: o evento e escolhido por chegar a este desfecho, entao
    as asseracoes valem sempre. A versao com `if` ficava verde mesmo se a regra
    de cobertura quebrasse.
    """
    payload = _evento_com_desfecho(motor, eventos, fault, "defeito_sem_documentacao")
    corpo = cliente.post("/diagnosticar", json=payload).json()

    assert corpo["situacao"] == "defeito_sem_documentacao"
    assert corpo["fault"] == fault
    assert corpo["cobertura"]["coberto"] is False
    assert not corpo["instrucoes"]
    assert not corpo["trechos"]
    assert "cadastre" in corpo["mensagem"].lower()


def test_estado_operacional_nao_recebe_acao_corretiva(cliente, motor, eventos):
    """Desfecho 3: motor desligado e condicao de operacao, nao problema."""
    payload = _evento_com_desfecho(motor, eventos, "motor_desligado", "estado_operacional")
    corpo = cliente.post("/diagnosticar", json=payload).json()

    assert corpo["situacao"] == "estado_operacional"
    assert corpo["e_defeito"] is False
    assert not corpo["instrucoes"]


def test_padrao_desconhecido_nao_arrisca_palpite(cliente):
    """Desfecho 4: um classificador fechado responderia a classe menos errada."""
    corpo = cliente.post("/diagnosticar", json=EVENTO_ABSURDO).json()

    assert corpo["situacao"] == "padrao_desconhecido"
    assert corpo["fault"] is None
    assert not corpo["instrucoes"]


def test_resposta_carrega_as_evidencias_da_decisao(cliente):
    corpo = cliente.post("/diagnosticar", json=EVENTO_ABSURDO).json()
    assert corpo["similaridade"]["motivo"]
    assert corpo["similaridade"]["reconhecido"] is False


# --------------------------------------------------------------------------
# /diagnosticos
# --------------------------------------------------------------------------


def test_diagnosticos_registra_e_devolve_o_historico(cliente, motor, eventos):
    payload = _evento_com_desfecho(motor, eventos, "ventoinha", "defeito_sem_documentacao")
    emitido = cliente.post("/diagnosticar", json=payload).json()
    assert emitido["registro_id"] is not None

    corpo = cliente.get("/diagnosticos").json()
    assert corpo["por_situacao"].get("defeito_sem_documentacao") == 1
    assert corpo["registros"]
    registro = corpo["registros"][0]
    assert registro["id"] == emitido["registro_id"]
    assert registro["situacao"] == "defeito_sem_documentacao"
    assert registro["origem"] == "api"


def test_diagnosticos_filtra_por_situacao(cliente, motor, eventos):
    cliente.post("/diagnosticar", json=EVENTO_ABSURDO)
    payload = _evento_com_desfecho(motor, eventos, "ventoinha", "defeito_sem_documentacao")
    cliente.post("/diagnosticar", json=payload)

    corpo = cliente.get("/diagnosticos", params={"situacao": "padrao_desconhecido"}).json()
    assert corpo["registros"]
    assert {r["situacao"] for r in corpo["registros"]} == {"padrao_desconhecido"}


# --------------------------------------------------------------------------
# 503 sem artefatos
# --------------------------------------------------------------------------


def test_sem_artefatos_a_api_responde_503(monkeypatch):
    """Indices ausentes viram 503 com instrucao, nao stack trace.

    E o primeiro estado de quem acabou de clonar o repositorio: a resposta
    precisa dizer qual script rodar.
    """
    def falhar() -> MotorDiagnostico:
        raise FileNotFoundError("indice_similaridade.joblib")

    monkeypatch.setattr(api.MotorDiagnostico, "carregar", staticmethod(falhar))
    monkeypatch.setattr(api, "_motor", None)
    api.app.dependency_overrides.clear()

    with TestClient(api.app) as testclient:
        resposta = testclient.get("/saude")

    assert resposta.status_code == 503
    assert "scripts/ingest.py" in resposta.json()["detail"]


# --------------------------------------------------------------------------
# POST /documentos - travessia de caminho, tamanho e PDF invalido
# --------------------------------------------------------------------------


def pdf_valido(titulo: str = "Procedimento de Teste") -> bytes:
    documento = fitz.open()
    pagina = documento.new_page()
    pagina.insert_textbox(
        fitz.Rect(50, 50, 545, 790),
        f"{titulo}\n\n1. Objetivo\nDocumento gerado pelo teste automatizado.",
        fontsize=11,
        fontname="helv",
    )
    dados = documento.tobytes()
    documento.close()
    return dados


class MotorParaUpload:
    """Motor minimo: o alvo destes testes e o handler, nao o diagnostico.

    Usar o motor real aqui trocaria a base dele por uma base de teste, e o
    proximo teste do modulo receberia essa base pela metade.
    """

    def __init__(self) -> None:
        self.base: Any = None


@pytest.fixture
def cliente_upload(tmp_path: pytest.TempPathFactory, monkeypatch):
    """Isola docs_dir, knowledge_dir e index_dir numa area descartavel.

    Sem isso um teste de upload gravaria em data/docs e reindexaria a base
    entregue - foi exatamente o que aconteceu ao reproduzir o ataque na mao.
    """
    settings = load_settings()
    area = Path(str(tmp_path))
    docs = area / "docs"
    docs.mkdir()
    knowledge = area / "knowledge"
    knowledge.mkdir()
    indice = area / "index"
    indice.mkdir()

    paths = dataclasses.replace(
        settings.paths, docs_dir=docs, knowledge_dir=knowledge, index_dir=indice
    )
    monkeypatch.setattr(api, "load_settings", lambda: dataclasses.replace(settings, paths=paths))
    api.app.dependency_overrides[api.obter_motor] = lambda: MotorParaUpload()
    try:
        with TestClient(api.app, raise_server_exceptions=False) as testclient:
            yield testclient, docs
    finally:
        api.app.dependency_overrides.clear()


@pytest.mark.parametrize("nome", NOMES_MALICIOSOS)
def test_nome_de_arquivo_malicioso_e_recusado(cliente_upload, nome):
    """Travessia de caminho: o conteudo e um PDF valido de proposito.

    Se a recusa viesse so da validacao de conteudo, este caso passaria. Falhar
    aqui significa que a unica barreira e o formato do arquivo.
    """
    testclient, docs = cliente_upload
    resposta = testclient.post(
        "/documentos", files={"arquivo": (nome, pdf_valido(), "application/pdf")}
    )

    assert resposta.status_code == 400, f"{nome!r} foi aceito"
    assert list(docs.glob("*.pdf")) == [], f"{nome!r} gravou arquivo"


def test_travessia_nao_escreve_fora_do_diretorio_de_documentos(cliente_upload, tmp_path):
    """Confirma no disco que nada apareceu acima de docs_dir."""
    testclient, docs = cliente_upload
    testclient.post(
        "/documentos", files={"arquivo": ("../../evil.pdf", pdf_valido(), "application/pdf")}
    )

    assert not (docs.parent / "evil.pdf").exists()
    assert not (docs.parent.parent / "evil.pdf").exists()
    assert not Path("evil.pdf").exists()


def test_pdf_invalido_e_recusado_sem_envenenar_a_base(cliente_upload):
    """O defeito era gravar antes de validar: o arquivo ruim ficava no glob.

    A segunda metade do teste e a que importa - depois da recusa, o cadastro
    continua funcionando. Antes do conserto ele respondia 500 para sempre.
    """
    testclient, docs = cliente_upload

    ruim = testclient.post(
        "/documentos", files={"arquivo": ("Ruim.pdf", b"nao sou um pdf" * 50, "application/pdf")}
    )
    assert ruim.status_code == 400
    assert list(docs.glob("*.pdf")) == [], "o PDF invalido nao pode ficar em docs_dir"

    bom = testclient.post(
        "/documentos", files={"arquivo": ("DocBom.pdf", pdf_valido(), "application/pdf")}
    )
    assert bom.status_code == 200, "um envio ruim nao pode derrubar o cadastro"
    assert bom.json()["arquivo"] == "DocBom.pdf"
    assert [p.name for p in docs.glob("*.pdf")] == ["DocBom.pdf"]


def test_upload_acima_do_teto_e_recusado(cliente_upload, monkeypatch):
    testclient, docs = cliente_upload
    monkeypatch.setattr(cadastro_mod, "TAMANHO_MAXIMO_BYTES", 1024)

    resposta = testclient.post(
        "/documentos", files={"arquivo": ("Grande.pdf", b"A" * 5000, "application/pdf")}
    )

    assert resposta.status_code == 413
    assert list(docs.glob("*.pdf")) == []


def test_documento_valido_e_cadastrado_e_fica_consultavel(cliente_upload):
    testclient, docs = cliente_upload
    resposta = testclient.post(
        "/documentos", files={"arquivo": ("DocNovo.pdf", pdf_valido(), "application/pdf")}
    )

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["arquivo"] == "DocNovo.pdf"
    assert corpo["documentos"] == 1
    assert corpo["trechos"] > 0
    assert (docs / "DocNovo.pdf").exists()


def test_reenviar_o_mesmo_nome_indexa_o_conteudo_novo(cliente_upload):
    """Corrigir um procedimento cadastrado tem de valer de verdade.

    O cache de extracao e por nome de arquivo: sem invalidacao, o reenvio
    respondia 200 e mantinha o texto antigo indexado, sem sinal nenhum para o
    operador.
    """
    testclient, docs = cliente_upload

    primeiro = testclient.post(
        "/documentos",
        files={"arquivo": ("Doc7.pdf", pdf_valido("Versao Errada"), "application/pdf")},
    )
    assert primeiro.status_code == 200

    segundo = testclient.post(
        "/documentos",
        files={"arquivo": ("Doc7.pdf", pdf_valido("Versao Corrigida"), "application/pdf")},
    )
    assert segundo.status_code == 200

    extraidos = extract_all(docs, Path(str(docs.parent / "knowledge")))
    texto = extraidos[0].texto
    assert "Versao Corrigida" in texto
    assert "Versao Errada" not in texto


def test_um_pdf_corrompido_no_diretorio_nao_derruba_a_reindexacao(tmp_path):
    """Segunda barreira, para o arquivo ruim que chegou por fora da API.

    Mora neste modulo porque e a garantia que mantem o cadastro de procedimentos
    de pe: sem ela, um PDF corrompido copiado na mao para data/docs faz toda
    reindexacao seguinte falhar, na API e na tela.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Bom.pdf").write_bytes(pdf_valido("Bom"))
    (docs / "Ruim.pdf").write_bytes(b"nao sou pdf" * 40)

    extraidos = extract_all(docs, tmp_path / "cache")

    assert [e.nome for e in extraidos] == ["Bom"]


def test_o_handler_de_upload_nao_roda_no_event_loop():
    """OCR a 25 s por pagina num `async def` deixaria /saude mudo por minutos.

    Sendo sincrono, o FastAPI despacha ao threadpool e o resto da API continua
    respondendo durante o cadastro.
    """
    import inspect

    assert not inspect.iscoroutinefunction(api.enviar_documento)
