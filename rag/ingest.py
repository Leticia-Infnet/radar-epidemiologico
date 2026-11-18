"""Download dos manuais do MS listados em `rag/fontes.yaml` e conversão em blocos de texto.

Roda no CI (workflow `index.yml`, por `rag.index`). Parsing com Docling, que preserva
tabelas e marca títulos de seção e a página de cada item; PyMuPDF é AGPL e está vetado. Os
PDFs ficam num diretório temporário, nunca no repositório.

Comandos auxiliares, só com o `pypdf` (`uv run --with pypdf --with pyyaml ...`):

    python -m rag.ingest --sumario <id>             # sumário com a página do PDF de cada entrada
    python -m rag.ingest --procurar <id> "termo"    # páginas em que o termo aparece
"""

import argparse
import hashlib
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from common.config import get_settings
from rag.chunk import Bloco, limpar
from rag.texto import normalizar

log = logging.getLogger(__name__)

FONTES = Path(__file__).with_name("fontes.yaml")
TENTATIVAS = 5


@dataclass(frozen=True)
class Manual:
    """Um documento do corpus, como descrito em `fontes.yaml`."""

    id: str
    titulo: str
    doencas: tuple[str, ...]
    edicao: str
    url: str
    paginas: tuple[int, int] | None = None  # páginas do PDF, a partir de 1, inclusivas
    ocr: bool = False  # páginas digitalizadas como imagem


def carregar_fontes(caminho: Path = FONTES) -> list[Manual]:
    """Lê a lista de manuais do YAML."""
    import yaml

    itens = yaml.safe_load(caminho.read_text(encoding="utf-8"))["manuais"]
    manuais = []
    for item in itens:
        paginas = item.get("paginas")
        manuais.append(
            Manual(
                id=item["id"],
                titulo=item["titulo"],
                doencas=tuple(item["doencas"]),
                edicao=str(item["edicao"]),
                url=item["url"],
                paginas=(int(paginas[0]), int(paginas[1])) if paginas else None,
                ocr=bool(item.get("ocr", False)),
            )
        )
    ids = [m.id for m in manuais]
    if len(set(ids)) != len(ids):
        raise ValueError(f"ids repetidos em {caminho}")
    return manuais


def baixar(manual: Manual, pasta: Path, cliente: httpx.Client | None = None) -> tuple[Path, str]:
    """Baixa o PDF para `pasta` e devolve o caminho e o SHA-256 do arquivo."""
    s = get_settings()
    cliente = cliente or httpx.Client(
        headers={"User-Agent": s.user_agent}, timeout=120.0, follow_redirects=True
    )
    destino = pasta / f"{manual.id}.pdf"
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            resumo = hashlib.sha256()
            with cliente.stream("GET", manual.url) as resposta, destino.open("wb") as arquivo:
                resposta.raise_for_status()
                for parte in resposta.iter_bytes():
                    resumo.update(parte)
                    arquivo.write(parte)
            with destino.open("rb") as arquivo:
                if arquivo.read(5) != b"%PDF-":
                    raise ValueError(f"{manual.url} não devolveu um PDF")
            log.info("%s: %d bytes baixados.", manual.id, destino.stat().st_size)
            return destino, resumo.hexdigest()
        except (httpx.HTTPError, ValueError) as exc:
            if tentativa == TENTATIVAS:
                raise
            espera = 10 * tentativa
            log.warning("%s: falha (%s); nova tentativa em %d s.", manual.id, exc, espera)
            time.sleep(espera)
    raise AssertionError("inalcançável")


def converter(caminho: Path, manual: Manual) -> list[Bloco]:
    """Converte o PDF com o Docling em blocos (títulos, parágrafos e tabelas) com a página."""
    # Importados aqui: o Docling puxa torch e só existe no grupo `indexacao`.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # OCR só nos manuais com páginas em imagem (`ocr: true` em fontes.yaml): nos demais,
    # só deixaria a conversão mais lenta. O RapidOCR roda no onnxruntime e o modelo
    # multilíngue padrão (PP-OCRv6) lê os acentos do português. Com OCR, o texto com
    # codificação quebrada também é refeito.
    opcoes = PdfPipelineOptions(
        do_ocr=manual.ocr,
        do_table_structure=True,
        ocr_options=RapidOcrOptions(replace_broken_text_layer=True),
    )
    conversor = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opcoes)}
    )
    paginas = manual.paginas or (1, sys.maxsize)
    documento = conversor.convert(caminho, page_range=paginas).document
    blocos = blocos_docling(documento, manual.id)
    log.info(
        "%s: %d blocos, páginas %d a %d.",
        manual.id,
        len(blocos),
        min((b.pagina for b in blocos), default=0),
        max((b.pagina for b in blocos), default=0),
    )
    return blocos


def blocos_docling(documento, manual_id: str) -> list[Bloco]:
    """Percorre o `DoclingDocument` em ordem de leitura e gera os blocos."""
    from docling_core.types.doc import DocItemLabel, TableItem, TextItem

    ignorados = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.PICTURE}
    titulos = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
    blocos: list[Bloco] = []
    pagina = 1
    for item, _nivel in documento.iterate_items():
        if item.label in ignorados:
            continue
        if getattr(item, "prov", None):
            pagina = item.prov[0].page_no
        if isinstance(item, TableItem):
            texto = item.export_to_markdown(doc=documento).strip()
            tipo = "tabela"
        elif isinstance(item, TextItem):
            texto = limpar(item.text)
            tipo = "titulo" if item.label in titulos else "texto"
            if item.label == DocItemLabel.LIST_ITEM and texto:
                texto = f"- {texto}"
        else:
            continue
        if texto:
            blocos.append(Bloco(manual_id, pagina, tipo, texto))
    return blocos


def _manual(manual_id: str) -> Manual:
    manuais = {m.id: m for m in carregar_fontes()}
    if manual_id not in manuais:
        raise SystemExit(f"id desconhecido: {manual_id}; opções: {', '.join(manuais)}")
    return manuais[manual_id]


def _sumario(manual: Manual, caminho: Path) -> None:
    from pypdf import PdfReader

    leitor = PdfReader(caminho)
    print(f"{manual.titulo}: {len(leitor.pages)} páginas")

    def percorrer(itens, nivel: int) -> None:
        for item in itens:
            if isinstance(item, list):
                percorrer(item, nivel + 1)
                continue
            pagina = leitor.get_destination_page_number(item) + 1
            print(f"{'  ' * nivel}p. {pagina:>4}  {item.title}")

    percorrer(leitor.outline, 0)


def _procurar(manual: Manual, caminho: Path, termo: str) -> None:
    from pypdf import PdfReader

    alvo = normalizar(termo)
    for numero, pagina in enumerate(PdfReader(caminho).pages, start=1):
        texto = normalizar(pagina.extract_text() or "")
        posicao = texto.find(alvo)
        if posicao >= 0:
            contexto = texto[max(0, posicao - 80) : posicao + len(alvo) + 80]
            print(f"p. {numero:>4}  …{contexto}…")


def main() -> None:
    """Comandos auxiliares para montar `fontes.yaml` e `avaliacao.yaml`."""
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--sumario", metavar="ID", help="Sumário do PDF com as páginas.")
    grupo.add_argument("--procurar", nargs=2, metavar=("ID", "TERMO"), help="Busca um termo.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    manual = _manual(args.sumario or args.procurar[0])
    with tempfile.TemporaryDirectory() as tmp:
        caminho, _ = baixar(manual, Path(tmp))
        if args.sumario:
            _sumario(manual, caminho)
        else:
            _procurar(manual, caminho, args.procurar[1])


if __name__ == "__main__":
    main()
