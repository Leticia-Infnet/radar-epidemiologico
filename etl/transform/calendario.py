"""Calendário de semanas epidemiológicas (domingo a sábado).

A semana 1 é a que contém o dia 4 de janeiro, ou seja, a primeira semana com ao menos
quatro dias no ano novo. Alguns anos têm 53 semanas (por exemplo, 2014, 2020 e 2025).
"""

from datetime import date, timedelta

import pandas as pd


def inicio_ano_epi(ano: int) -> date:
    """Domingo que abre a semana epidemiológica 1 do ano."""
    quatro_jan = date(ano, 1, 4)
    return quatro_jan - timedelta(days=(quatro_jan.weekday() + 1) % 7)


def semanas_no_ano(ano: int) -> int:
    """Quantidade de semanas epidemiológicas do ano (52 ou 53)."""
    return (inicio_ano_epi(ano + 1) - inicio_ano_epi(ano)).days // 7


def calendario(ano_inicial: int, ano_final: int) -> pd.DataFrame:
    """Todas as semanas epidemiológicas do intervalo, com a data de início de cada uma."""
    linhas = [
        {
            "ano_epi": ano,
            "semana_epi": semana,
            "inicio": pd.Timestamp(inicio_ano_epi(ano) + timedelta(weeks=semana - 1)),
        }
        for ano in range(ano_inicial, ano_final + 1)
        for semana in range(1, semanas_no_ano(ano) + 1)
    ]
    return pd.DataFrame(linhas)
