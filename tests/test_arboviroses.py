import struct
from contextlib import contextmanager
from datetime import date

import polars as pl
import pytest

from common.config import Settings
from etl import pipeline
from etl.extract import datasus
from etl.extract.datasus import ArquivoSinan, escolher_arquivos_sinan, ler_colunas_dbf
from etl.transform.aggregate import (
    CAMPOS_SINAN,
    contar_por_semana,
    semanas,
    series_semanais,
)
from etl.transform.calendario import semana_epidemiologica
from etl.transform.populacao import normalizar_populacao
from etl.transform.schemas import CasosSemanais, Populacao, ResumoSemanal

FINAIS = "/SINAN/DADOS/FINAIS"
PRELIM = "/SINAN/DADOS/PRELIM"


def _dbf(caminho, campos: list[tuple[str, int]], linhas: list[tuple[str, ...]], apagadas=()):
    """Grava um DBF (dBase III) mínimo, só com campos de texto."""
    tamanho_registro = 1 + sum(n for _, n in campos)
    tamanho_cabecalho = 32 + 32 * len(campos) + 1
    cabecalho = struct.pack(
        "<BBBBIHH20x", 3, 126, 11, 4, len(linhas), tamanho_cabecalho, tamanho_registro
    )
    descritores = b"".join(
        struct.pack("<11sc4xBB14x", nome.encode(), b"C", n, 0) for nome, n in campos
    )
    registros = b"".join(
        (b"*" if i in apagadas else b" ")
        + b"".join(v.encode("latin-1").ljust(n) for v, (_, n) in zip(linha, campos, strict=True))
        for i, linha in enumerate(linhas)
    )
    caminho.write_bytes(cabecalho + descritores + b"\r" + registros + b"\x1a")
    return caminho


def test_ler_colunas_dbf_recorta_campos_e_ignora_apagados(tmp_path):
    dbf = _dbf(
        tmp_path / "t.dbf",
        [("NOME", 8), ("ID_MN_RESI", 6), ("SEM_PRI", 6)],
        [("ANA", "260960", "202401"), ("JOSÉ", " 26096", "20240\xe7"), ("X", "1", "2")],
        apagadas={2},
    )
    tabela = ler_colunas_dbf(dbf, ("SEM_PRI", "ID_MN_RESI"))
    assert tabela.columns == ["SEM_PRI", "ID_MN_RESI"]
    assert tabela.rows() == [("202401", "260960"), ("20240?", "26096")]
    with pytest.raises(KeyError, match="CLASSI_FIN"):
        ler_colunas_dbf(dbf, ("CLASSI_FIN",))


def test_escolhe_um_arquivo_por_ano_preferindo_o_final():
    finais = [f"{FINAIS}/DENGBR{a}.dbc" for a in ("09", "10", "11")] + [f"{FINAIS}/CHIKBR11.dbc"]
    prelim = [f"{PRELIM}/DENGBR11.dbc", f"{PRELIM}/DENGBR12.dbc", f"{PRELIM}/LEIAME.txt"]
    arquivos = escolher_arquivos_sinan(finais, prelim, "dengue", 2010)
    assert arquivos == [
        ArquivoSinan("dengue", 2010, f"{FINAIS}/DENGBR10.dbc", False),
        ArquivoSinan("dengue", 2011, f"{FINAIS}/DENGBR11.dbc", False),
        ArquivoSinan("dengue", 2012, f"{PRELIM}/DENGBR12.dbc", True),
    ]
    with pytest.raises(LookupError, match="2011"):
        escolher_arquivos_sinan(finais[:2], prelim[1:], "dengue", 2010)
    with pytest.raises(LookupError):
        escolher_arquivos_sinan(finais, prelim, "zika", 2010)


@pytest.mark.parametrize(
    ("dia", "esperado"),
    [
        (date(2024, 1, 6), (2024, 1)),
        (date(2023, 12, 31), (2024, 1)),  # domingo que abre o ano epidemiológico de 2024
        (date(2023, 12, 30), (2023, 52)),
        (date(2026, 1, 1), (2025, 53)),
        (date(2026, 10, 27), (2026, 43)),
    ],
)
def test_semana_epidemiologica(dia, esperado):
    assert semana_epidemiologica(dia) == esperado


def _registros(linhas):
    return pl.DataFrame(linhas, schema=list(CAMPOS_SINAN), orient="row")


def test_contar_por_semana():
    registros = _registros(
        [
            ("260960", "202401", "10", "1"),
            ("2609600", "202401", "", "2"),  # 7 dígitos; óbito pelo agravo
            ("260960", "202401", "5", "2"),  # descartado: o óbito também sai
            ("260960", "202401", "2", ""),  # descartado (SINAN NET)
            ("", "202402", "8", "3"),  # sem residência: conta, com residência nula
            ("26X960", "202402", "10", ""),
            ("260960", "202353", "10", ""),  # 2023 só tem 52 semanas
            ("260960", "2024", "10", ""),
            ("260960", None, "10", ""),
            ("260960", "202444", "10", ""),  # começa depois do corte
        ]
    )
    contagem, descartes = contar_por_semana(registros, corte=date(2024, 10, 1))
    assert descartes == {
        "descartado": 2,
        "semana inválida": 3,
        "semana posterior ao corte": 1,
    }
    assert sorted(contagem.rows(), key=str) == sorted(
        [("260960", 2024, 1, 2, 1), (None, 2024, 2, 2, 0)], key=str
    )


def test_semanas_vai_ate_a_semana_do_corte():
    calendario = semanas(2025, date(2026, 1, 13))
    assert calendario.height == 53 + 2
    assert calendario.row(0) == (2025, 1, date(2024, 12, 29))
    assert calendario.row(-1) == (2026, 2, date(2026, 1, 11))


def _contagem(linhas):
    return pl.DataFrame(
        linhas,
        schema={
            "residencia": pl.String,
            "ano_epi": pl.Int64,
            "semana_epi": pl.Int64,
            "casos_provaveis": pl.Int64,
            "obitos": pl.Int64,
        },
        orient="row",
    )


def test_series_semanais_completam_zeros_somam_arquivos_e_marcam_incompletas():
    arquivo_2025 = _contagem([("260960", 2025, 52, 3, 0), ("261160", 2025, 53, 1, 0)])
    # O arquivo de 2026 traz casos de semanas de 2025 e de quem não está no dicionário.
    arquivo_2026 = _contagem(
        [
            ("260960", 2025, 52, 2, 1),
            ("260960", 2026, 2, 4, 0),
            ("260000", 2026, 1, 5, 0),  # código de UF sem município
            (None, 2026, 1, 7, 0),
            ("260960", 2024, 52, 9, 0),  # antes do ano inicial
        ]
    )
    municipal, resumo, totais = series_semanais(
        [arquivo_2025, arquivo_2026], ["260960", "261160"], "dengue", 2025, date(2026, 1, 13)
    )
    CasosSemanais.validate(municipal)
    ResumoSemanal.validate(resumo)
    assert municipal.height == 2 * 55
    olinda = municipal.filter(pl.col("codigo_ibge6") == "260960")
    assert olinda["casos_provaveis"].sum() == 9
    assert olinda.filter(pl.col("semana_epi") == 52, pl.col("ano_epi") == 2025).row(0)[6:8] == (
        5,
        1,
    )
    assert set(municipal["uf_codigo"]) == {"26"}
    # Corte em 13/01/2026 e 4 semanas incompletas na dengue: a partir de 16/12/2025.
    incompletas = olinda.filter("semana_incompleta")["inicio_semana"].to_list()
    assert incompletas == [
        date(2025, 12, 21),
        date(2025, 12, 28),
        date(2026, 1, 4),
        date(2026, 1, 11),
    ]

    def total(local):
        return resumo.filter(pl.col("local") == local)["casos_provaveis"].sum()

    assert total("26") == 3 + 1 + 2 + 4 + 5
    assert total("BR") == total("26") + 7
    assert total("35") == 0
    assert resumo.filter(pl.col("local") == "35").height == 55
    assert totais == {
        "casos_provaveis": 22,
        "obitos": 1,
        "casos_sem_municipio_no_dicionario": 12,
        "casos_sem_uf_valida": 7,
    }


def test_normalizar_populacao():
    tabela = normalizar_populacao(
        [
            {"codigo_municipio": "260960", "populacao_estimada_ibge_2022": 349976},
            {"codigo_municipio": 120001, "populacao_estimada_ibge_2022": "14021"},
            {"codigo_municipio": "999", "populacao_estimada_ibge_2022": 10},
            {"codigo_municipio": "120005", "populacao_estimada_ibge_2022": None},
        ]
    )
    Populacao.validate(tabela)
    assert tabela.rows() == [("120001", 14021, 2022), ("260960", 349976, 2022)]


def test_arboviroses_abaixo_do_piso_nao_publica(monkeypatch, tmp_path):
    arquivo = ArquivoSinan("dengue", 2026, f"{PRELIM}/DENGBR26.dbc", True)
    registros = _registros([("260960", "202601", "10", "")])
    publicacoes = []
    monkeypatch.setattr(pipeline, "ANOS_INICIAIS", {"dengue": 2026})
    monkeypatch.setattr(pipeline, "_municipios_publicados", lambda: ["260960"])
    monkeypatch.setattr(pipeline.datasus, "arquivos_sinan", lambda d, a: [arquivo])
    monkeypatch.setattr(
        pipeline.datasus,
        "ler_sinan",
        lambda arq, campos: (registros, pipeline.datetime(2026, 2, 1, tzinfo=pipeline.UTC)),
    )
    monkeypatch.setattr(pipeline, "publicar_pasta", lambda *a, **k: publicacoes.append(a))
    with pytest.raises(pipeline.ExtracaoIncompleta, match="dengue"):
        pipeline.executar_arboviroses(publicar=True)
    assert publicacoes == []


def test_arboviroses_grava_particoes_por_uf(monkeypatch, tmp_path):
    arquivo = ArquivoSinan("dengue", 2026, f"{PRELIM}/DENGBR26.dbc", True)
    registros = _registros([("260960", "202601", "10", ""), ("120001", "202602", "10", "2")])
    monkeypatch.setattr(pipeline, "ANOS_INICIAIS", {"dengue": 2026})
    monkeypatch.setitem(pipeline.MINIMO_CASOS, "dengue", 1)
    monkeypatch.setattr(pipeline.datasus, "arquivos_sinan", lambda d, a: [arquivo])
    monkeypatch.setattr(
        pipeline.datasus,
        "ler_sinan",
        lambda arq, campos: (registros, pipeline.datetime(2026, 2, 1, tzinfo=pipeline.UTC)),
    )
    linhas, metadados = pipeline.gerar_arboviroses(tmp_path, ["120001", "260960"])
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet")) == [
        "doenca=dengue/uf=12/casos.parquet",
        "doenca=dengue/uf=26/casos.parquet",
        "resumo.parquet",
    ]
    acre = pl.read_parquet(tmp_path / "doenca=dengue/uf=12/casos.parquet")
    assert acre["obitos"].sum() == 1
    assert linhas == 2 * 5  # semanas 1 a 5 de 2026 (corte em 01/02)
    assert metadados["dengue"]["ultima_semana"] == "2026-05"
    assert metadados["dengue"]["corte"] == "2026-02-01"
    assert metadados["dengue"]["arquivos"]["DENGBR26.dbc"]["preliminar"] is True


class _FtpQueCai:
    """FTP simulado que derruba a conexão depois de alguns bytes, como o do DATASUS."""

    def __init__(self, conteudo: bytes, quedas: int, bytes_por_conexao: int):
        self.conteudo, self.quedas, self.por_conexao = conteudo, quedas, bytes_por_conexao
        self.retomadas: list[int | None] = []

    def voidcmd(self, comando):
        return "213 20261027101500"

    def size(self, caminho):
        return len(self.conteudo)

    def retrbinary(self, comando, callback, rest=None):
        self.retomadas.append(rest)
        inicio = rest or 0
        if self.quedas:
            self.quedas -= 1
            callback(self.conteudo[inicio : inicio + self.por_conexao])
            raise ConnectionResetError(104, "Connection reset by peer")
        callback(self.conteudo[inicio:])


def test_baixar_retoma_de_onde_parou(monkeypatch, tmp_path):
    ftp = _FtpQueCai(bytes(range(256)) * 40, quedas=3, bytes_por_conexao=1000)

    @contextmanager
    def conexao(settings):
        yield ftp

    monkeypatch.setattr(datasus, "_conexao", conexao)
    monkeypatch.setattr(datasus._baixar.retry, "sleep", lambda _: None)
    destino = tmp_path / "DENGBR24.dbc"
    publicado = datasus._baixar("/FINAIS/DENGBR24.dbc", destino, Settings())
    assert destino.read_bytes() == ftp.conteudo
    assert ftp.retomadas == [None, 1000, 2000, 3000]
    assert publicado.isoformat() == "2026-10-27T10:15:00+00:00"
