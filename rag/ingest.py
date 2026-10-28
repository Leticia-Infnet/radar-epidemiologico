"""Download dos manuais do MS listados em `rag/fontes.yaml` e conversão para texto.

Roda só no CI (workflow `index.yml`). Parsing com Docling ou pypdf; PyMuPDF é AGPL e
está vetado. Os PDFs ficam no diretório temporário do runner, nunca no repositório.
"""


def main() -> None:
    """Baixa cada manual, extrai texto por página e preserva os cabeçalhos de seção."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
