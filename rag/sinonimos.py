"""Expansão da pergunta com sinônimos técnicos (`rag/sinonimos.yaml`).

O usuário escreve "grávida", "coceira", "remédio"; os manuais, "gestante", "prurido",
"medicamento". A busca por palavras (BM25) não liga esses termos, e o e5-small nem sempre
os aproxima. Acrescentar os termos técnicos à pergunta antes das duas buscas trouxe o
trecho certo em mais perguntas coloquiais (medido em 2026-11-17).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from rag.texto import normalizar

ARQUIVO = Path(__file__).with_name("sinonimos.yaml")


@dataclass(frozen=True)
class Sinonimo:
    padrao: re.Pattern[str]
    termos: str


def carregar_sinonimos(caminho: Path = ARQUIVO) -> list[Sinonimo]:
    """Lê e compila as regras do YAML."""
    import yaml

    regras = yaml.safe_load(caminho.read_text(encoding="utf-8"))["sinonimos"]
    return [Sinonimo(re.compile(r["padrao"]), str(r["termos"]).strip()) for r in regras]


def expandir(pergunta: str, sinonimos: list[Sinonimo]) -> str:
    """A pergunta com os termos técnicos das regras que casam com ela, sem repetir."""
    texto = normalizar(pergunta)
    extras = list(dict.fromkeys(s.termos for s in sinonimos if s.padrao.search(texto)))
    return f"{pergunta} {' '.join(extras)}" if extras else pergunta
