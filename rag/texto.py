"""Normalização e tokenização de texto em português, comuns ao filtro de alarme, ao chunking
e à busca lexical (BM25)."""

import re
import unicodedata

# Palavras muito frequentes, sem acento (o texto é normalizado antes da comparação).
STOPWORDS: frozenset[str] = frozenset(
    """
    a ao aos apos aquela aquelas aquele aqueles aquilo as ate cada com como da das de dela
    delas dele deles depois deve devo do dos e ela elas ele eles em entre era essa essas
    esse esses esta estao estas estava este estes estou eu fica ficar foi for foram ha isso
    isto ja la lhe lhes mais mas me mesmo meu minha muito na nas nem no nos nossa nosso num
    numa o onde os ou para pela pelas pelo pelos pode pois por porque posso quais qual
    quando que quem sao se seja sem sendo ser seu seus so sobre sua suas tambem te tem tenho
    ter teu tipo tua um uma umas uns voce voces vos
    """.split()  # noqa: SIM905 (lista longa fica legível em texto)
)


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos e com espaços colapsados."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


def tokenizar(texto: str) -> list[str]:
    """Palavras normalizadas, sem stopwords e sem tokens de um caractere."""
    return [
        t for t in re.findall(r"[a-z0-9]+", normalizar(texto)) if len(t) > 1 and t not in STOPWORDS
    ]
