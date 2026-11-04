"""Arquivos do CNES no FTP do DATASUS: profissionais (PF) e tabela de ocupações (CBO).

A API da DEMAS não serve para isto: `/assistencia-a-saude/cnes-profissionais` devolve um
registro por competência e para em 2017. O FTP publica um arquivo por UF e mês,
`PF<UF><AAMM>.dbc`, com um vínculo profissional por linha. O `.dbc` é um DBF comprimido;
`datasus-dbc` (MIT) o descomprime e `dbfread` (MIT) lê o DBF. O `pyreaddbc`, mais usado,
é AGPL e fica de fora (AGENTS.md, seção 6).
"""

import ftplib
import logging
import re
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import datasus_dbc
import dbfread
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from common.config import Settings, get_settings

log = logging.getLogger(__name__)

UFS = (
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",  # fmt: skip
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",  # fmt: skip
)
_ARQUIVO_PF = re.compile(r"^PF([A-Z]{2})(\d{4})\.dbc$", re.IGNORECASE)

# Ocupações (6 posições) e famílias (4 posições) da CBO, com acentos; a `CBO.dbf` do
# mesmo pacote vem sem acentos.
_TABELAS_CBO = ("DBF/CBO_02.dbf", "DBF/CBO_FAM.dbf")

_tentar_de_novo = retry(
    retry=retry_if_exception_type((ftplib.Error, OSError, EOFError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=5, max=60),
    reraise=True,
)


@_tentar_de_novo
def _listar(settings: Settings) -> list[str]:
    with ftplib.FTP(settings.datasus_ftp_host, timeout=settings.datasus_ftp_timeout_s) as ftp:
        ftp.login()
        return ftp.nlst(settings.datasus_cnes_pf_dir)


@_tentar_de_novo
def _baixar(caminho_remoto: str, destino: Path, settings: Settings) -> None:
    with (
        ftplib.FTP(settings.datasus_ftp_host, timeout=settings.datasus_ftp_timeout_s) as ftp,
        destino.open("wb") as arquivo,
    ):
        ftp.login()
        ftp.retrbinary(f"RETR {caminho_remoto}", arquivo.write)


def competencia_completa(nomes: list[str], ufs: tuple[str, ...] = UFS) -> str:
    """Competência `AAMM` mais recente que já tem arquivo PF para todas as UFs.

    O DATASUS publica as UFs aos poucos; usar a mais recente de cada uma misturaria meses.
    """
    por_competencia: dict[str, set[str]] = {}
    for nome in nomes:
        achado = _ARQUIVO_PF.match(Path(nome).name)
        if achado:
            uf, aamm = achado.group(1).upper(), achado.group(2)
            por_competencia.setdefault(aamm, set()).add(uf)
    completas = [aamm for aamm, presentes in por_competencia.items() if presentes >= set(ufs)]
    if not completas:
        raise LookupError("Nenhuma competência do CNES tem arquivo PF para todas as UFs.")
    # AAMM ordena como texto a partir de 2000.
    return max(completas)


def competencia_pf(settings: Settings | None = None) -> str:
    """Consulta o FTP e devolve a competência completa mais recente (`AAMM`)."""
    return competencia_completa(_listar(settings or get_settings()))


def ler_dbc(caminho: Path, campos: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    """Descomprime um `.dbc` num diretório temporário e devolve só os campos pedidos."""
    with tempfile.TemporaryDirectory() as tmp:
        dbf = Path(tmp) / caminho.with_suffix(".dbf").name
        datasus_dbc.decompress(str(caminho), str(dbf))
        for registro in dbfread.DBF(dbf, encoding="latin-1"):
            yield tuple(str(registro.get(c) or "").strip() for c in campos)


def profissionais_uf(
    uf: str, aamm: str, campos: tuple[str, ...], settings: Settings | None = None
) -> Iterator[tuple[str, ...]]:
    """Baixa o arquivo PF de uma UF e competência e percorre seus vínculos.

    O arquivo só existe num diretório temporário enquanto é lido.
    """
    s = settings or get_settings()
    nome = f"PF{uf}{aamm}.dbc"
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / nome
        _baixar(f"{s.datasus_cnes_pf_dir}/{nome}", local, s)
        log.info("%s: %.1f MB baixados.", nome, local.stat().st_size / 1e6)
        yield from ler_dbc(local, campos)


def ocupacoes_cbo(settings: Settings | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """Nomes das ocupações (código de 6 posições) e das famílias (4 posições) da CBO.

    Vêm da `TAB_CNES.zip` do DATASUS. A família dá nome às ocupações que faltam na tabela.
    """
    s = settings or get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        pacote = pasta / "TAB_CNES.zip"
        _baixar(s.datasus_cnes_tabelas, pacote, s)
        tabelas = []
        with zipfile.ZipFile(pacote) as z:
            for interno in _TABELAS_CBO:
                dbf = pasta / Path(interno).name
                dbf.write_bytes(z.read(interno))
                tabelas.append(ler_tabela_cbo(dbf))
        return tabelas[0], tabelas[1]


def ler_tabela_cbo(dbf: Path) -> dict[str, str]:
    """Lê uma tabela de conversão do CNES (campos `CHAVE` e `DS_REGRA`)."""
    tabela = {}
    for registro in dbfread.DBF(dbf, encoding="latin-1"):
        codigo = str(registro["CHAVE"] or "").strip()
        nome = str(registro["DS_REGRA"] or "").strip()
        if codigo and nome:
            tabela[codigo] = nome
    return tabela
