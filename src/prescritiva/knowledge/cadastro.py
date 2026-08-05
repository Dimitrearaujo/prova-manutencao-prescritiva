"""Cadastro de documento orientativo, compartilhado entre a API e o painel.

O endpoint HTTP e o painel Streamlit fazem a mesma coisa - recebem um PDF novo
e o colocam para prescrever - mas o painel reimplementava o upload por conta
propria e pulava tres protecoes que a API tinha: nome de arquivo validado
contra travessia de caminho, teto de tamanho do envio, e invalidacao do cache
de extracao. Sem a terceira, reenviar um PDF corrigido pelo painel com o mesmo
nome respondia sucesso e continuava prescrevendo pelo texto antigo - o
contrario do que o cadastro existe para resolver.

Esta e agora a unica porta de entrada para gravar um documento em docs_dir e
reconstruir a base de conhecimento. API e painel chamam `cadastrar_documento`
e traduzem `CadastroInvalido` para a sua propria forma de erro.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import fitz

from prescritiva.config import Settings
from prescritiva.knowledge.extract import extract_all
from prescritiva.knowledge.store import BaseConhecimento

LOGGER = logging.getLogger(__name__)

# O nome vem de um campo sob controle de quem envia (cabecalho multipart na
# API, nome do arquivo escolhido no seletor do painel). Allowlist em vez de
# lista de proibidos porque a variedade de formas de escrever um caminho
# ("..", "C:/", UNC, separador invertido, byte nulo) e grande demais para
# enumerar sem deixar buraco.
NOME_DOCUMENTO = re.compile(r"[A-Za-z0-9_.-]{1,80}\.pdf", re.IGNORECASE)

# Teto de tamanho para que um envio absurdo nao ocupe o disco da estacao nem
# entre na reindexacao. Nao impede o recebimento - isso e papel do servidor na
# frente, quando existe um - mas garante que nada alem deste tamanho chegue a
# docs_dir.
TAMANHO_MAXIMO_BYTES = 25 * 1024 * 1024
BLOCO_BYTES = 1024 * 1024


class CadastroInvalido(Exception):
    """Erro de validacao no cadastro: nome, tamanho ou PDF ilegivel.

    Carrega o `status_code` HTTP que o caso pede (400 ou 413) para a API
    repassar direto; o painel Streamlit ignora o codigo e usa so a mensagem.
    """

    def __init__(self, mensagem: str, *, status_code: int = 400) -> None:
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.status_code = status_code


@dataclass(frozen=True)
class ResultadoCadastro:
    nome: str
    base: BaseConhecimento


def nome_seguro(filename: str | None, docs_dir: Path) -> str:
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
        raise CadastroInvalido("Envie apenas o nome do arquivo, sem caminho.")
    if not NOME_DOCUMENTO.fullmatch(nome):
        raise CadastroInvalido(
            "Nome de arquivo invalido: use letras, numeros, ponto, hifen ou "
            "sublinhado, terminando em .pdf."
        )
    if not (docs_dir / nome).resolve().is_relative_to(docs_dir.resolve()):
        raise CadastroInvalido("Nome de arquivo invalido.")
    return nome


def _gravar_em_blocos(arquivo: BinaryIO, destino: Path) -> None:
    """Copia o envio para o disco em blocos, abortando ao passar do teto.

    Ler o arquivo inteiro em memoria antes de gravar deixa o tamanho do envio
    decidir quanta RAM o processo que atende a planta consome.
    """
    total = 0
    with destino.open("wb") as saida:
        while bloco := arquivo.read(BLOCO_BYTES):
            total += len(bloco)
            if total > TAMANHO_MAXIMO_BYTES:
                raise CadastroInvalido(
                    f"Arquivo acima do limite de {TAMANHO_MAXIMO_BYTES // (1024 * 1024)} MB.",
                    status_code=413,
                )
            saida.write(bloco)


def _exigir_pdf_legivel(caminho: Path) -> None:
    """Recusa o que o extrator nao conseguiria abrir depois.

    A validacao acontece enquanto o arquivo ainda esta na area temporaria. Se
    ela rodasse depois da gravacao em docs_dir, o arquivo ruim ficaria no
    diretorio e toda reindexacao seguinte morreria nele.
    """
    try:
        with fitz.open(caminho) as documento:
            paginas = documento.page_count
    except Exception as erro:  # noqa: BLE001 - qualquer falha de leitura vira recusa, nunca 500
        LOGGER.warning("cadastro recusado, PDF ilegivel: %s", erro)
        raise CadastroInvalido("O arquivo enviado nao e um PDF legivel.") from erro
    if paginas < 1:
        raise CadastroInvalido("O PDF enviado nao tem paginas.")


def cadastrar_documento(arquivo: BinaryIO, filename: str | None, settings: Settings) -> ResultadoCadastro:
    """Valida, grava, invalida o cache de extracao e reindexa a base.

    E o unico ponto que escreve em docs_dir e reconstroi `BaseConhecimento`.
    Quem chama - API ou painel - so decide o que fazer com o resultado.
    """
    docs_dir = settings.paths.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    nome = nome_seguro(filename, docs_dir)

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
    # responde sucesso e o sistema segue prescrevendo pela versao errada, que e
    # o oposto do que este cadastro existe para resolver.
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

    return ResultadoCadastro(nome=nome, base=base)
