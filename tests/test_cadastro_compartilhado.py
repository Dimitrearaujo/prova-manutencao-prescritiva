"""Testes de `prescritiva.knowledge.cadastro`, a rotina que API e painel chamam.

`tests/test_api.py` ja cobre o endpoint por cima (nome malicioso, teto de
tamanho, PDF ilegivel, reenvio invalida o cache). Este modulo testa a mesma
regra por baixo, direto na funcao compartilhada, com um objeto `BytesIO`
puro - a mesma interface que o `UploadedFile` do Streamlit oferece - em vez do
`UploadFile` do FastAPI. E a garantia de que o painel, que nao tem suite de
testes propria, recebe as mesmas protecoes sem reimplementa-las.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import fitz
import pytest

from prescritiva.config import load_settings
from prescritiva.knowledge import cadastro as cadastro_mod
from prescritiva.knowledge.cadastro import CadastroInvalido, cadastrar_documento
from prescritiva.knowledge.extract import extract_all


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


@pytest.fixture
def settings_isolados(tmp_path: Path):
    """Mesma isolacao de `cliente_upload` em test_api.py, sem passar pela API.

    Um teste que gravasse em data/docs reindexaria a base entregue de verdade.
    """
    base = load_settings()
    docs = tmp_path / "docs"
    docs.mkdir()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    indice = tmp_path / "index"
    indice.mkdir()
    paths = dataclasses.replace(
        base.paths, docs_dir=docs, knowledge_dir=knowledge, index_dir=indice
    )
    return dataclasses.replace(base, paths=paths)


def _bytes_io(nome: str, conteudo: bytes) -> io.BytesIO:
    """Simula o `UploadedFile` do Streamlit: um `BytesIO` com `.name`."""
    arquivo = io.BytesIO(conteudo)
    arquivo.name = nome  # type: ignore[attr-defined]
    return arquivo


def test_arquivo_valido_e_gravado_e_indexado(settings_isolados):
    resultado = cadastrar_documento(
        _bytes_io("DocPainel.pdf", pdf_valido()), "DocPainel.pdf", settings_isolados
    )

    assert resultado.nome == "DocPainel.pdf"
    assert resultado.base.documentos
    assert (settings_isolados.paths.docs_dir / "DocPainel.pdf").exists()


@pytest.mark.parametrize(
    "nome",
    ["../../evil.pdf", "..\\..\\evil.pdf", "sem_extensao", "script.pdf.exe"],
)
def test_nome_invalido_e_recusado_sem_gravar(settings_isolados, nome):
    with pytest.raises(CadastroInvalido) as excinfo:
        cadastrar_documento(_bytes_io(nome, pdf_valido()), nome, settings_isolados)

    assert excinfo.value.status_code == 400
    assert list(settings_isolados.paths.docs_dir.glob("*.pdf")) == []


def test_pdf_ilegivel_e_recusado_sem_envenenar_a_base(settings_isolados):
    with pytest.raises(CadastroInvalido) as excinfo:
        cadastrar_documento(
            _bytes_io("Ruim.pdf", b"nao sou um pdf" * 50), "Ruim.pdf", settings_isolados
        )

    assert excinfo.value.status_code == 400
    assert list(settings_isolados.paths.docs_dir.glob("*.pdf")) == []


def test_acima_do_teto_e_recusado(settings_isolados, monkeypatch):
    monkeypatch.setattr(cadastro_mod, "TAMANHO_MAXIMO_BYTES", 1024)

    with pytest.raises(CadastroInvalido) as excinfo:
        cadastrar_documento(_bytes_io("Grande.pdf", b"A" * 5000), "Grande.pdf", settings_isolados)

    assert excinfo.value.status_code == 413
    assert list(settings_isolados.paths.docs_dir.glob("*.pdf")) == []


def test_reenviar_o_mesmo_nome_pelo_painel_indexa_o_conteudo_novo(settings_isolados):
    """A regressao que motivou a unificacao: cache sobrevivendo ao reenvio.

    Antes, o painel escrevia o PDF e reindexava sem tocar no cache de
    extracao. Reenviar `Doc7.pdf` corrigido continuava servindo o texto da
    versao antiga, porque o cache era indexado por nome de arquivo.
    """
    cadastrar_documento(
        _bytes_io("Doc7.pdf", pdf_valido("Versao Errada")), "Doc7.pdf", settings_isolados
    )
    cadastrar_documento(
        _bytes_io("Doc7.pdf", pdf_valido("Versao Corrigida")), "Doc7.pdf", settings_isolados
    )

    extraidos = extract_all(settings_isolados.paths.docs_dir, settings_isolados.paths.knowledge_dir)
    texto = extraidos[0].texto
    assert "Versao Corrigida" in texto
    assert "Versao Errada" not in texto


# --------------------------------------------------------------------------
# Chave de acesso (#17) - so exige quando settings.cadastro tem chave_acesso
# --------------------------------------------------------------------------


def _com_chave(settings, chave):
    """Mesmos settings isolados, com `cadastro.chave_acesso` configurada."""
    return dataclasses.replace(settings, cadastro={**settings.cadastro, "chave_acesso": chave})


def test_sem_chave_configurada_cadastro_continua_aberto(settings_isolados):
    """Estado de desenvolvimento: nenhuma instalacao nova fica travada de saida."""
    assert not settings_isolados.cadastro.get("chave_acesso")

    resultado = cadastrar_documento(_bytes_io("Doc.pdf", pdf_valido()), "Doc.pdf", settings_isolados)

    assert resultado.nome == "Doc.pdf"


def test_chave_configurada_recusa_sem_chave(settings_isolados):
    settings = _com_chave(settings_isolados, "segredo-planta")

    with pytest.raises(CadastroInvalido) as excinfo:
        cadastrar_documento(_bytes_io("Doc.pdf", pdf_valido()), "Doc.pdf", settings)

    assert excinfo.value.status_code == 401
    assert list(settings.paths.docs_dir.glob("*.pdf")) == []


def test_chave_configurada_recusa_chave_errada(settings_isolados):
    settings = _com_chave(settings_isolados, "segredo-planta")

    with pytest.raises(CadastroInvalido) as excinfo:
        cadastrar_documento(
            _bytes_io("Doc.pdf", pdf_valido()), "Doc.pdf", settings, chave_apresentada="chave-errada"
        )

    assert excinfo.value.status_code == 401


def test_chave_configurada_aceita_chave_certa(settings_isolados):
    settings = _com_chave(settings_isolados, "segredo-planta")

    resultado = cadastrar_documento(
        _bytes_io("Doc.pdf", pdf_valido()), "Doc.pdf", settings, chave_apresentada="segredo-planta"
    )

    assert resultado.nome == "Doc.pdf"


# --------------------------------------------------------------------------
# Portao de conteudo minimo (#16) - PDF que abre mas nao rendeu texto o
# bastante para virar procedimento consultavel
# --------------------------------------------------------------------------


def pdf_quase_vazio() -> bytes:
    """PDF valido, uma pagina, sem texto nenhum - o caso do escaneamento ruim."""
    documento = fitz.open()
    documento.new_page()
    dados = documento.tobytes()
    documento.close()
    return dados


def test_pdf_sem_conteudo_util_e_recusado_sem_entrar_na_base(settings_isolados):
    with pytest.raises(CadastroInvalido) as excinfo:
        cadastrar_documento(
            _bytes_io("Vazio.pdf", pdf_quase_vazio()), "Vazio.pdf", settings_isolados
        )

    assert excinfo.value.status_code == 400
    assert "caracteres" in excinfo.value.mensagem
    # Nem o PDF nem o cache de extracao podem sobreviver a recusa - senao o
    # proximo cadastro encontraria um documento fantasma no glob.
    assert list(settings_isolados.paths.docs_dir.glob("*.pdf")) == []
    assert list(settings_isolados.paths.knowledge_dir.glob("*.json")) == []


def _pdf_com_texto_curto(texto: str) -> bytes:
    documento = fitz.open()
    pagina = documento.new_page()
    pagina.insert_textbox(fitz.Rect(50, 50, 545, 100), texto, fontsize=11, fontname="helv")
    dados = documento.tobytes()
    documento.close()
    return dados


def test_limiar_de_conteudo_e_configuravel(settings_isolados):
    """Mesmo PDF: recusado no limiar padrao, aceito quando o limiar baixa.

    O ponto do teste e que o limiar vem de `settings.cadastro`, nao de um
    numero fixo no codigo - "Doc curto" tem menos de 40 caracteres (o padrao),
    mas passa de sobra num limiar de 5.
    """
    texto_curto = "Doc curto"  # 9 caracteres, abaixo do minimo padrao (40)

    with pytest.raises(CadastroInvalido):
        cadastrar_documento(
            _bytes_io("Curto.pdf", _pdf_com_texto_curto(texto_curto)), "Curto.pdf", settings_isolados
        )

    settings_permissivo = dataclasses.replace(
        settings_isolados, cadastro={**settings_isolados.cadastro, "min_chars_por_pagina": 5}
    )
    resultado = cadastrar_documento(
        _bytes_io("Curto.pdf", _pdf_com_texto_curto(texto_curto)), "Curto.pdf", settings_permissivo
    )
    assert resultado.nome == "Curto.pdf"
