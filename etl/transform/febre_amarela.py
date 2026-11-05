"""Casos humanos confirmados de febre amarela da DEMAS, agregados por município e mês.

A fonte é diferente da das outras arboviroses (AGENTS.md, seção 7.1):

- são **casos confirmados**, não prováveis;
- o município é o **local provável de infecção** (`cod_mun_lpi`), em geral área de mata,
  não a residência;
- a vigilância conta por **período de monitoramento**, de julho a junho, porque a
  transmissão se concentra entre dezembro e maio.

A tabela publicada traz só contagens: idade e sexo dos casos não saem desta etapa.
"""

import re
from collections import Counter
from datetime import date, datetime
from typing import Any

import polars as pl

from etl.transform.aggregate import UF_CODIGOS
from etl.transform.normalize import ausente, codigo_ibge6

MOTIVOS_DE_DESCARTE = ("registro repetido", "sem data nem período")
_PERIODO = re.compile(r"^(\d{4})/(\d{4})$")

ESQUEMA = {
    "periodo": pl.String,
    "ano_inicio_periodo": pl.Int64,
    "ano": pl.Int64,
    "mes": pl.Int64,
    "codigo_ibge6": pl.String,
    "uf_codigo": pl.String,
    "casos": pl.Int64,
    "obitos": pl.Int64,
}


def periodo_de_monitoramento(dia: date) -> str:
    """Período de julho a junho que contém a data: `2024-11-03` → `2024/2025`."""
    inicio = dia.year if dia.month >= 7 else dia.year - 1
    return f"{inicio}/{inicio + 1}"


def _inteiro(valor: Any) -> int | None:
    if ausente(valor):
        return None
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return None


def _data(valor: Any) -> date | None:
    if ausente(valor):
        return None
    try:
        return datetime.strptime(str(valor).strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def normalizar_febre_amarela(
    registros: list[dict[str, Any]],
) -> tuple[pl.DataFrame, dict[str, int], date | None]:
    """Agrega os casos por período, ano, mês, município e UF de infecção.

    A DEMAS repete linhas idênticas (32 de 2.913 em 2026-11); só a primeira conta. O
    período vem da data de início dos sintomas e, sem ela, do campo `monitoramento_is`.
    Devolve a tabela, os descartes por motivo e a data de início de sintomas mais recente.
    """
    descartes = dict.fromkeys(MOTIVOS_DE_DESCARTE, 0)
    vistos: set[tuple] = set()
    contagem: Counter[tuple] = Counter()
    obitos: Counter[tuple] = Counter()
    mais_recente: date | None = None
    for registro in registros:
        chave_registro = tuple(sorted((k, str(v)) for k, v in registro.items()))
        if chave_registro in vistos:
            descartes["registro repetido"] += 1
            continue
        vistos.add(chave_registro)

        inicio_sintomas = _data(registro.get("dt_is"))
        if inicio_sintomas:
            periodo = periodo_de_monitoramento(inicio_sintomas)
            ano, mes = inicio_sintomas.year, inicio_sintomas.month
            if mais_recente is None or inicio_sintomas > mais_recente:
                mais_recente = inicio_sintomas
        else:
            texto = (
                ""
                if ausente(registro.get("monitoramento_is"))
                else str(registro["monitoramento_is"])
            )
            achado = _PERIODO.match(texto.strip())
            if not achado:
                descartes["sem data nem período"] += 1
                continue
            periodo = achado.group(0)
            ano, mes = _inteiro(registro.get("ano_is")), _inteiro(registro.get("mes_is"))
            if mes is not None and not 1 <= mes <= 12:
                mes = None

        municipio = codigo_ibge6(registro.get("cod_mun_lpi"))
        uf = _inteiro(registro.get("cod_uf_lpi"))
        uf_codigo = f"{uf:02d}" if uf is not None else (municipio[:2] if municipio else None)
        if uf_codigo not in UF_CODIGOS:
            uf_codigo = None
        chave = (periodo, int(periodo[:4]), ano, mes, municipio, uf_codigo)
        contagem[chave] += 1
        obitos[chave] += str(registro.get("obito", "")).strip().upper() == "SIM"

    campos = ("periodo", "ano_inicio_periodo", "ano", "mes", "codigo_ibge6", "uf_codigo")
    linhas = [
        {**dict(zip(campos, chave, strict=True)), "casos": n, "obitos": obitos[chave]}
        for chave, n in contagem.items()
    ]
    tabela = pl.DataFrame(linhas, schema=ESQUEMA).sort(
        "ano_inicio_periodo", "ano", "mes", "codigo_ibge6", nulls_last=True
    )
    return tabela, descartes, mais_recente
