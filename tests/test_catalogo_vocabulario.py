"""O catalogo nao pode ganhar sinonimo inventado.

`termos_pergunta` existe porque o radical corta em seis letras e separa palavras
que o tecnico usa como sinonimo: "balanceamento" vira "balanc" e
"desbalanceamento" vira "desbal"; "cocked rotor" e o nome que o proprio Doc6 usa
no titulo e nao casa com "rotor inclinado". Sem esses termos o portao do chat
recusava pergunta legitima, que e a falha pior das duas.

O risco do remedio e virar caca ao rato: cada vez que alguem encontra um jeito
novo de perguntar, acrescenta-se mais uma palavra, e o catalogo passa a refletir
a imaginacao de quem editou em vez do acervo. Foi assim que o conserto anterior
deste portao falhou - ele cobriu exatamente as frases da auditoria e mais nenhuma.

Estes testes fecham essa porta com duas regras, e as duas olham para os
documentos, nao para o gosto de quem escreve o YAML:

    defeito COBERTO      todo termo tem que aparecer no texto do documento que o
                         cobre. Se a palavra nao esta la, ela nao e como aquele
                         procedimento chama aquele defeito.

    defeito SEM DOCUMENTO  nenhum termo pode aparecer no escopo de documento
                         algum. Se aparecesse, o defeito estaria coberto, e o
                         lugar de resolver isso e a cobertura, nao o vocabulario.
"""

from __future__ import annotations

import json

import pytest

from prescritiva.config import load_catalog, load_settings
from prescritiva.knowledge.store import BaseConhecimento
from prescritiva.text import normalize, stems


@pytest.fixture(scope="module")
def base() -> BaseConhecimento:
    indice = load_settings().paths.index_dir
    if not (indice / "base_conhecimento.json").exists():
        pytest.skip("execute scripts/build_knowledge.py antes")
    return BaseConhecimento.carregar(indice)


@pytest.fixture(scope="module")
def catalogo() -> dict:
    return load_catalog()["faults"]


@pytest.fixture(scope="module")
def texto_por_documento() -> dict[str, str]:
    """Texto integral de cada procedimento, normalizado."""
    diretorio = load_settings().paths.knowledge_dir
    arquivos = sorted(diretorio.glob("Doc*.json"))
    if not arquivos:
        pytest.skip("execute scripts/extract_docs.py antes")
    return {
        arquivo.stem: normalize(json.loads(arquivo.read_text(encoding="utf-8"))["texto"])
        for arquivo in arquivos
    }


def _com_termos(catalogo: dict) -> list[tuple[str, dict]]:
    return [(fault, e) for fault, e in catalogo.items() if e.get("termos_pergunta")]


def test_existe_ao_menos_um_termo_para_testar(catalogo):
    """Guarda contra o teste virar vacuo se alguem esvaziar o campo."""
    assert _com_termos(catalogo), "nenhum defeito declara termos_pergunta"


def test_termo_de_defeito_coberto_existe_no_documento_que_o_cobre(
    base, catalogo, texto_por_documento
):
    for fault, entrada in _com_termos(catalogo):
        cobertura = base.cobertura(
            entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
        )
        if not cobertura.coberto:
            continue
        corpo = texto_por_documento[cobertura.documento]
        for termo in entrada["termos_pergunta"]:
            assert normalize(termo) in corpo, (
                f'"{termo}" foi declarado como forma de nomear {fault}, mas nao aparece '
                f"em {cobertura.documento}, que e o procedimento que cobre esse defeito. "
                "Termo de defeito coberto sai do documento, nao da cabeca de quem edita."
            )


def test_termo_de_defeito_sem_procedimento_nao_aparece_em_nenhum_escopo(base, catalogo):
    for fault, entrada in _com_termos(catalogo):
        cobertura = base.cobertura(
            entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
        )
        if cobertura.coberto:
            continue
        for termo in entrada["termos_pergunta"]:
            radicais = stems(termo)
            for documento in base.documentos:
                escopo = stems(documento.escopo) | stems(documento.titulo)
                assert not radicais <= escopo, (
                    f'"{termo}" foi declarado como forma de nomear {fault}, que nao tem '
                    f"procedimento - mas o escopo de {documento.nome} contem esse termo. "
                    "Se o escopo contem, o defeito esta coberto e o problema e de "
                    "cobertura, nao de vocabulario."
                )


def test_termo_nao_pode_ancorar_defeito_de_outro_componente(base, catalogo):
    """Um termo nao pode casar com o termo-chave de OUTRO defeito.

    O caso concreto que motivou a regra: "excentricidade" sozinha. O Doc5 tem uma
    secao "3.1 Excentricidade" sobre a excentricidade DA POLIA, defeito que ele
    cobre. Declarar a palavra solta como forma de nomear o rotor excentrico faria
    o sistema recusar uma pergunta legitima sobre polia. O defeito e o par
    fenomeno + componente; a palavra sozinha nao identifica nada.
    """
    chaves = {fault: stems(e["termo_chave"]) for fault, e in catalogo.items()}
    for fault, entrada in _com_termos(catalogo):
        for termo in entrada["termos_pergunta"]:
            radicais = stems(termo)
            for outro, chave_do_outro in chaves.items():
                if outro == fault:
                    continue
                assert not chave_do_outro <= radicais, (
                    f'"{termo}", declarado para {fault}, contem o termo-chave de {outro}. '
                    "Uma pergunta com esse termo passaria a nomear os dois defeitos."
                )
