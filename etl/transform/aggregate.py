"""Agregação das notificações do SINAN por município de residência e semana epidemiológica.

Decisões (AGENTS.md, seção 12, validadas em 2026-11-04):

- município de referência: **residência** (`ID_MN_RESI`), como nos boletins do MS;
- semana de referência: **início dos sintomas** (`SEM_PRI`);
- definição de caso: **casos prováveis**, isto é, notificações não descartadas. Os
  arquivos de dengue do FTP já vêm sem os descartados; nos outros agravos, eles saem aqui;
- recorte: Brasil inteiro, a partir do ano inicial de cada agravo.
"""

from datetime import date, timedelta

import polars as pl

from etl.transform.calendario import inicio_ano_epi, semana_epidemiologica, semanas_no_ano

CAMPOS_SINAN = ("ID_MN_RESI", "SEM_PRI", "CLASSI_FIN", "EVOLUCAO")

# Classificação final: 2 (SINAN NET) e 5 (SINAN Online) são descartados; os demais
# códigos, inclusive em branco (em investigação), contam como caso provável.
DESCARTADOS = ("2", "5")
OBITO_PELO_AGRAVO = "2"  # campo EVOLUCAO

# Códigos IBGE das 27 UFs.
UF_CODIGOS = frozenset(
    [
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "31",
        "32",
        "33",
        "35",
        "41",
        "42",
        "43",
        "50",
        "51",
        "52",
        "53",
    ]
)
BRASIL = "BR"

# Semanas mais recentes que o atraso de notificação deixa incompletas. O p90 do atraso
# foi de 6 dias na dengue e de 35 na zika (notebook de análise exploratória).
SEMANAS_INCOMPLETAS = {"dengue": 4, "chikungunya": 4, "zika": 8}

MOTIVOS_DE_DESCARTE = ("descartado", "semana inválida", "semana posterior ao corte")

_CONTAGEM = {
    "residencia": pl.String,
    "ano_epi": pl.Int64,
    "semana_epi": pl.Int64,
    "casos_provaveis": pl.Int64,
    "obitos": pl.Int64,
}


def _inicio_da_semana(anos: list[int]) -> pl.Expr:
    inicios = {ano: inicio_ano_epi(ano) for ano in anos}
    return pl.col("ano_epi").replace_strict(inicios, return_dtype=pl.Date) + pl.duration(
        weeks=pl.col("semana_epi") - 1
    )


def contar_por_semana(registros: pl.DataFrame, corte: date) -> tuple[pl.DataFrame, dict[str, int]]:
    """Casos prováveis e óbitos de um arquivo do SINAN por residência e semana.

    `registros` traz os campos de `CAMPOS_SINAN` como texto. A residência é normalizada
    para o código IBGE de 6 dígitos, ou nulo quando ausente ou inválida: esses casos
    ainda contam no total do Brasil. Semanas que começam depois de `corte` (a data de
    publicação do arquivo) são erros de digitação e saem, contadas em `descartes`.
    """
    descartes = dict.fromkeys(MOTIVOS_DE_DESCARTE, 0)
    descartado = pl.col("CLASSI_FIN").is_in(DESCARTADOS)
    descartes["descartado"] = int(registros.select(descartado.sum()).item() or 0)
    base = registros.filter(~descartado.fill_null(False)).with_columns(
        residencia=pl.when(pl.col("ID_MN_RESI").str.contains(r"^\d{6,7}$"))
        .then(pl.col("ID_MN_RESI").str.slice(0, 6))
        .otherwise(None),
        ano_epi=pl.col("SEM_PRI").str.slice(0, 4).cast(pl.Int64, strict=False),
        semana_epi=pl.col("SEM_PRI").str.slice(4, 2).cast(pl.Int64, strict=False),
        obito=(pl.col("EVOLUCAO") == OBITO_PELO_AGRAVO).fill_null(False),
    )

    anos = [a for a in base["ano_epi"].unique().to_list() if a is not None and 1900 < a < 2100]
    limite = pl.col("ano_epi").replace_strict(
        {a: semanas_no_ano(a) for a in anos}, default=0, return_dtype=pl.Int64
    )
    semana_valida = (
        pl.col("SEM_PRI").str.contains(r"^\d{6}$").fill_null(False)
        & (pl.col("semana_epi") >= 1)
        & (pl.col("semana_epi") <= limite)
    )
    validas = base.filter(semana_valida)
    descartes["semana inválida"] = base.height - validas.height
    validas = validas.with_columns(inicio_semana=_inicio_da_semana(anos))
    no_prazo = validas.filter(pl.col("inicio_semana") <= corte)
    descartes["semana posterior ao corte"] = validas.height - no_prazo.height

    contagem = no_prazo.group_by("residencia", "ano_epi", "semana_epi").agg(
        casos_provaveis=pl.len().cast(pl.Int64),
        obitos=pl.col("obito").sum().cast(pl.Int64),
    )
    return contagem.select(list(_CONTAGEM)).cast(_CONTAGEM), descartes


def semanas(ano_inicial: int, corte: date) -> pl.DataFrame:
    """Todas as semanas epidemiológicas do ano inicial até a semana que contém o corte."""
    ano_final, semana_final = semana_epidemiologica(corte)
    linhas = [
        (ano, semana)
        for ano in range(ano_inicial, ano_final + 1)
        for semana in range(1, semanas_no_ano(ano) + 1)
        if (ano, semana) <= (ano_final, semana_final)
    ]
    tabela = pl.DataFrame(
        linhas, schema={"ano_epi": pl.Int64, "semana_epi": pl.Int64}, orient="row"
    )
    return tabela.with_columns(
        inicio_semana=_inicio_da_semana(list(range(ano_inicial, ano_final + 1)))
    )


def _completar(
    contagem: pl.DataFrame, chaves: pl.DataFrame, coluna: str, calendario: pl.DataFrame
) -> pl.DataFrame:
    """Cruza cada chave com todas as semanas e preenche com zero as que não têm casos."""
    grade = chaves.join(calendario, how="cross")
    return grade.join(contagem, on=[coluna, "ano_epi", "semana_epi"], how="left").with_columns(
        pl.col("casos_provaveis").fill_null(0), pl.col("obitos").fill_null(0)
    )


def series_semanais(
    contagens: list[pl.DataFrame],
    municipios: list[str],
    doenca: str,
    ano_inicial: int,
    corte: date,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """Séries semanais completas (com zeros) por município e por UF e Brasil.

    `contagens` são as saídas de `contar_por_semana` de todos os arquivos do agravo; uma
    mesma semana pode aparecer em mais de um arquivo e é somada. `municipios` são os
    códigos de 6 dígitos do dicionário: casos de residência fora dele entram só nos
    totais da UF (se o código da UF for válido) e do Brasil. `corte` é a data de
    publicação do arquivo mais recente; as últimas `SEMANAS_INCOMPLETAS[doenca]` semanas
    antes dele ficam marcadas como incompletas.
    """
    soma = (
        pl.concat(contagens)
        .filter(pl.col("ano_epi") >= ano_inicial)
        .group_by("residencia", "ano_epi", "semana_epi")
        .agg(pl.col("casos_provaveis").sum(), pl.col("obitos").sum())
    )
    calendario = semanas(ano_inicial, corte)
    limite_incompleto = corte - timedelta(weeks=SEMANAS_INCOMPLETAS[doenca])
    marcar = [
        pl.lit(doenca).alias("doenca"),
        (pl.col("inicio_semana") > limite_incompleto).alias("semana_incompleta"),
    ]

    por_municipio = (
        _completar(
            soma.rename({"residencia": "codigo_ibge6"}),
            pl.DataFrame({"codigo_ibge6": sorted(municipios)}),
            "codigo_ibge6",
            calendario,
        )
        .with_columns(*marcar, uf_codigo=pl.col("codigo_ibge6").str.slice(0, 2))
        .select(
            "doenca",
            "codigo_ibge6",
            "uf_codigo",
            "ano_epi",
            "semana_epi",
            "inicio_semana",
            "casos_provaveis",
            "obitos",
            "semana_incompleta",
        )
        .sort("codigo_ibge6", "ano_epi", "semana_epi")
    )

    uf = pl.col("residencia").str.slice(0, 2)
    por_uf = (
        soma.filter(uf.is_in(UF_CODIGOS))
        .group_by(uf.alias("local"), "ano_epi", "semana_epi")
        .agg(pl.col("casos_provaveis").sum(), pl.col("obitos").sum())
    )
    brasil = soma.group_by("ano_epi", "semana_epi").agg(
        pl.lit(BRASIL).alias("local"), pl.col("casos_provaveis").sum(), pl.col("obitos").sum()
    )
    resumo = (
        _completar(
            pl.concat([por_uf, brasil.select(por_uf.columns)]),
            pl.DataFrame({"local": [BRASIL, *sorted(UF_CODIGOS)]}),
            "local",
            calendario,
        )
        .with_columns(*marcar)
        .select(
            "doenca",
            "local",
            "ano_epi",
            "semana_epi",
            "inicio_semana",
            "casos_provaveis",
            "obitos",
            "semana_incompleta",
        )
        .sort("local", "ano_epi", "semana_epi")
    )

    no_periodo = soma.join(
        calendario.select("ano_epi", "semana_epi"), on=["ano_epi", "semana_epi"], how="semi"
    )
    total = int(no_periodo["casos_provaveis"].sum())
    fora_do_dicionario = total - int(por_municipio["casos_provaveis"].sum())
    sem_uf = total - int(no_periodo.filter(uf.is_in(UF_CODIGOS))["casos_provaveis"].sum())
    resumo_contagens = {
        "casos_provaveis": total,
        "obitos": int(no_periodo["obitos"].sum()),
        "casos_sem_municipio_no_dicionario": fora_do_dicionario,
        "casos_sem_uf_valida": sem_uf,
    }
    return por_municipio, resumo, resumo_contagens
