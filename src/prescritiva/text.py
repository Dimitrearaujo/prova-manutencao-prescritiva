"""Normalizacao de texto para busca.

O Doc1 chega por OCR e o modelo perde acentuacao ("correcao", "diagnostico").
Normalizar acento e caixa nos dois lados da busca torna esse ruido irrelevante,
em vez de exigir um OCR perfeito.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+")

# Palavras muito frequentes em procedimento tecnico em portugues: mantidas fora
# do indice para nao dominarem o BM25.
STOPWORDS = frozenset(
    """a ao aos as com como da das de do dos e em entre essa esse esta este eu
    ha isso isto na nas no nos o os ou para pela pelas pelo pelos por qual quando
    que se sem seu seus sua suas tem ter um uma umas uns ate apos sobre""".split()
)


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(value: str) -> str:
    return strip_accents(value).lower()


def tokenize(value: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(normalize(value))
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


_TAMANHO_RADICAL = 6


def singular(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith(("ais", "eis", "ois", "uis")):
        return token[:-2] + "l"
    if token.endswith(("oes", "aes")):
        return token[:-3] + "ao"
    if token.endswith("ns"):
        return token[:-2] + "m"
    if token.endswith("s"):
        return token[:-1]
    return token


def stem(token: str) -> str:
    """Plural do portugues, depois truncamento do prefixo.

    Sao dois passos porque um so nao resolve. O truncamento sozinho separaria
    "polia" de "polias" (5 e 6 letras) e o documento de polias deixaria de cobrir
    a falha de polia. A regra de plural sozinha nao une "desalinhado" com
    "desalinhamento" nem "excentrico" com "excentricidade", que sao derivacoes.

    O corte em 6 e o que separa "correia" de "correcao": as duas palavras so
    divergem na sexta letra, e cortar antes disso faria a falha de correia casar
    com todos os seis procedimentos, ja que todos tem "Correcao" no titulo.
    """
    return singular(token)[:_TAMANHO_RADICAL]


def stems(value: str, *, drop_stopwords: bool = True) -> set[str]:
    return {stem(token) for token in tokenize(value, drop_stopwords=drop_stopwords)}


def collapse_whitespace(value: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", value)).strip()
