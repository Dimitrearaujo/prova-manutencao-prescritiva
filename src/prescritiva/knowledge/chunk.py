"""Fatiamento dos procedimentos em trechos recuperaveis.

Os seis documentos seguem a mesma estrutura numerada ("1. Objetivo",
"3.1 Excentricidade"). Cortar respeitando esse indice mantem cada trecho dentro
de um assunto so, e carregar o titulo da secao junto do texto faz o trecho
chegar ao modelo ja contextualizado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CABECALHO = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")

# Passos numerados dentro de um procedimento tem a mesma forma de um titulo de
# secao ("3. Trocar lubrificante."), mas terminam em pontuacao. O titulo real
# nao termina. Sem esta distincao, cada passo abriria uma secao de uma linha e o
# trecho recuperado chegaria ao leitor sem o contexto do procedimento.
_FIM_DE_FRASE = (".", ";", ":", ",")


@dataclass
class Trecho:
    documento: str
    secao: str
    ordem: int
    texto: str

    @property
    def identificador(self) -> str:
        return f"{self.documento}#{self.ordem}"


def _secoes(texto: str) -> list[tuple[str, str]]:
    secao_atual = "Introducao"
    linhas_atuais: list[str] = []
    resultado: list[tuple[str, str]] = []

    for linha in texto.splitlines():
        limpa = linha.strip()
        if _CABECALHO.match(limpa) and len(limpa) < 90 and not limpa.endswith(_FIM_DE_FRASE):
            if linhas_atuais:
                resultado.append((secao_atual, "\n".join(linhas_atuais).strip()))
            secao_atual = limpa
            linhas_atuais = []
        else:
            linhas_atuais.append(linha)

    if linhas_atuais:
        resultado.append((secao_atual, "\n".join(linhas_atuais).strip()))
    return [(secao, corpo) for secao, corpo in resultado if corpo]


def _fatiar(corpo: str, tamanho: int, sobreposicao: int) -> list[str]:
    if len(corpo) <= tamanho:
        return [corpo]
    pedacos: list[str] = []
    inicio = 0
    while inicio < len(corpo):
        fim = inicio + tamanho
        if fim < len(corpo):
            # Preferir terminar em quebra de linha para nao partir um item de lista.
            quebra = corpo.rfind("\n", inicio + tamanho // 2, fim)
            if quebra != -1:
                fim = quebra
        pedacos.append(corpo[inicio:fim].strip())
        if fim >= len(corpo):
            break
        inicio = max(fim - sobreposicao, inicio + 1)
    return [p for p in pedacos if p]


def dividir(
    documento: str, texto: str, *, tamanho: int = 900, sobreposicao: int = 150
) -> list[Trecho]:
    trechos: list[Trecho] = []
    for secao, corpo in _secoes(texto):
        for pedaco in _fatiar(corpo, tamanho, sobreposicao):
            trechos.append(
                Trecho(
                    documento=documento,
                    secao=secao,
                    ordem=len(trechos),
                    texto=f"{secao}\n{pedaco}" if secao not in pedaco else pedaco,
                )
            )
    return trechos
