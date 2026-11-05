"""População por município, do cadastro de regiões de saúde da DEMAS."""

from typing import Any

import polars as pl

from etl.transform.normalize import codigo_ibge6

# O endpoint traz só a estimativa de 2022. A população por ano existe na DEMAS
# (`/vigilancia-e-meio-ambiente/svsa-populacao-residente`), mas aberta por idade e sexo:
# cerca de 900 mil linhas por ano, inviável de baixar a cada execução.
CAMPO_POPULACAO = "populacao_estimada_ibge_2022"
ANO_REFERENCIA = 2022


def normalizar_populacao(registros: list[dict[str, Any]]) -> pl.DataFrame:
    """Código IBGE de 6 dígitos e população; descarta registros sem um dos dois."""
    linhas = {}
    for registro in registros:
        codigo = codigo_ibge6(registro.get("codigo_municipio"))
        try:
            populacao = int(registro.get(CAMPO_POPULACAO))
        except (TypeError, ValueError):
            continue
        if codigo and populacao > 0:
            linhas[codigo] = populacao
    return pl.DataFrame(
        {
            "codigo_ibge6": list(linhas),
            "populacao": list(linhas.values()),
            "ano_referencia": [ANO_REFERENCIA] * len(linhas),
        },
        schema={"codigo_ibge6": pl.String, "populacao": pl.Int64, "ano_referencia": pl.Int64},
    ).sort("codigo_ibge6")
