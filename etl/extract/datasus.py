"""Arquivos do FTP do DATASUS: CNES (profissionais e ocupações) e SINAN (arboviroses).

CNES: a API da DEMAS não serve para profissionais (`/assistencia-a-saude/cnes-profissionais`
devolve um registro por competência e para em 2017). O FTP publica um arquivo por UF e
mês, `PF<UF><AAMM>.dbc`, com um vínculo profissional por linha.

SINAN: um arquivo nacional por agravo e ano, `<AGRAVO>BR<AA>.dbc`, em `FINAIS` (anos
encerrados) e `PRELIM` (ano corrente e anos ainda não encerrados). Os de dengue trazem só
casos prováveis: os descartados não são publicados (verificado em 2017 e 2026).

O `.dbc` é um DBF comprimido; `datasus-dbc` (MIT) o descomprime e `dbfread` (MIT) lê o
DBF. O `pyreaddbc`, mais usado, é AGPL e fica de fora (AGENTS.md, seção 6).
"""

import ftplib
import logging
import re
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import datasus_dbc
import dbfread
import numpy as np
import polars as pl
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
    "PA",
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
    "TO",
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
def _listar(diretorio: str, settings: Settings) -> list[str]:
    with ftplib.FTP(settings.datasus_ftp_host, timeout=settings.datasus_ftp_timeout_s) as ftp:
        ftp.login()
        return ftp.nlst(diretorio)


@_tentar_de_novo
def _data_de_modificacao(caminho_remoto: str, settings: Settings) -> datetime:
    with ftplib.FTP(settings.datasus_ftp_host, timeout=settings.datasus_ftp_timeout_s) as ftp:
        ftp.login()
        # Resposta no formato "213 AAAAMMDDhhmmss", em UTC.
        resposta = ftp.voidcmd(f"MDTM {caminho_remoto}")
    return datetime.strptime(resposta.split()[1][:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)


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
    s = settings or get_settings()
    return competencia_completa(_listar(s.datasus_cnes_pf_dir, s))


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


def ler_colunas_dbc(caminho: Path, campos: tuple[str, ...]) -> pl.DataFrame:
    """Descomprime um `.dbc` num diretório temporário e lê só os campos pedidos."""
    with tempfile.TemporaryDirectory() as tmp:
        dbf = Path(tmp) / caminho.with_suffix(".dbf").name
        datasus_dbc.decompress(str(caminho), str(dbf))
        return ler_colunas_dbf(dbf, campos)


def _texto_ascii(bruto: np.ndarray) -> pl.Series:
    # np.array já copiou o campo do arquivo para a memória; dá para alterar no lugar.
    octetos = bruto.view(np.uint8)
    octetos[octetos > 127] = ord("?")
    return pl.Series(bruto).cast(pl.Binary).cast(pl.String).str.strip_chars()


def ler_colunas_dbf(dbf: Path, campos: tuple[str, ...]) -> pl.DataFrame:
    """Lê só os campos pedidos de um DBF, como texto sem espaços nas bordas.

    Registros de DBF têm largura fixa: em vez de interpretar cada linha inteira (o que o
    `dbfread` faz a cerca de 75 µs por registro), os campos são recortados direto do
    arquivo com NumPy, cerca de 70 vezes mais rápido. Registros apagados (marcados com
    `*`) ficam de fora. Os campos lidos são códigos: bytes fora do ASCII (lixo de
    digitação em alguns arquivos) viram `?`, e o código fica inválido.
    """
    tabela = dbfread.DBF(dbf, load=False)
    posicoes, inicio = {}, 1  # o primeiro byte do registro marca exclusão
    for campo in tabela.fields:
        posicoes[campo.name] = (inicio, campo.length)
        inicio += campo.length
    faltando = set(campos) - set(posicoes)
    if faltando:
        raise KeyError(f"{dbf.name} não tem os campos {sorted(faltando)}.")
    n = tabela.header.numrecords
    if n == 0:
        return pl.DataFrame({c: pl.Series([], dtype=pl.String) for c in campos})
    formato = np.dtype(
        {
            "names": ["_apagado", *campos],
            "formats": ["S1", *(f"S{posicoes[c][1]}" for c in campos)],
            "offsets": [0, *(posicoes[c][0] for c in campos)],
            "itemsize": tabela.header.recordlen,
        }
    )
    registros = np.memmap(dbf, dtype=formato, mode="r", offset=tabela.header.headerlen, shape=(n,))
    colunas = {c: _texto_ascii(np.array(registros[c])) for c in campos}
    validos = pl.Series(np.array(registros["_apagado"]) != b"*")
    del registros
    return pl.DataFrame(colunas).filter(validos)


# Prefixo dos arquivos do SINAN por agravo.
AGRAVOS_SINAN = {"dengue": "DENG", "chikungunya": "CHIK", "zika": "ZIKA"}
_ARQUIVO_SINAN = re.compile(r"^([A-Z]{4})BR(\d{2})\.dbc$", re.IGNORECASE)


@dataclass(frozen=True)
class ArquivoSinan:
    """Arquivo nacional de um agravo e ano no FTP do DATASUS."""

    doenca: str
    ano: int
    caminho: str
    preliminar: bool


def escolher_arquivos_sinan(
    finais: list[str], preliminares: list[str], doenca: str, ano_inicial: int
) -> list[ArquivoSinan]:
    """Um arquivo por ano a partir de `ano_inicial`, preferindo o final ao preliminar.

    `finais` e `preliminares` são as listagens (caminhos completos) das pastas `FINAIS` e
    `PRELIM`. Levanta `LookupError` se faltar algum ano entre o inicial e o mais recente.
    """
    prefixo = AGRAVOS_SINAN[doenca]
    por_ano: dict[int, ArquivoSinan] = {}
    for caminhos, preliminar in ((preliminares, True), (finais, False)):
        for caminho in caminhos:
            achado = _ARQUIVO_SINAN.match(Path(caminho).name)
            if not achado or achado.group(1).upper() != prefixo:
                continue
            ano = 2000 + int(achado.group(2))
            if ano >= ano_inicial:
                # Os finais vêm depois e substituem o preliminar do mesmo ano.
                por_ano[ano] = ArquivoSinan(doenca, ano, caminho, preliminar)
    if not por_ano:
        raise LookupError(f"Nenhum arquivo de {doenca} a partir de {ano_inicial} no FTP.")
    faltando = sorted(set(range(ano_inicial, max(por_ano) + 1)) - set(por_ano))
    if faltando:
        raise LookupError(f"{doenca}: faltam no FTP os arquivos dos anos {faltando}.")
    return [por_ano[ano] for ano in sorted(por_ano)]


def arquivos_sinan(
    doenca: str, ano_inicial: int, settings: Settings | None = None
) -> list[ArquivoSinan]:
    """Consulta as pastas `FINAIS` e `PRELIM` e escolhe um arquivo por ano."""
    s = settings or get_settings()
    return escolher_arquivos_sinan(
        _listar(s.datasus_sinan_finais_dir, s),
        _listar(s.datasus_sinan_prelim_dir, s),
        doenca,
        ano_inicial,
    )


def ler_sinan(
    arquivo: ArquivoSinan, campos: tuple[str, ...], settings: Settings | None = None
) -> tuple[pl.DataFrame, datetime]:
    """Baixa o arquivo, lê os campos pedidos e devolve também a data de publicação.

    A data de publicação (modificação no FTP) é o corte dos dados do arquivo. O arquivo
    só existe num diretório temporário enquanto é lido.
    """
    s = settings or get_settings()
    publicado_em = _data_de_modificacao(arquivo.caminho, s)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(arquivo.caminho).name
        _baixar(arquivo.caminho, local, s)
        log.info("%s: %.1f MB baixados.", local.name, local.stat().st_size / 1e6)
        return ler_colunas_dbc(local, campos), publicado_em
