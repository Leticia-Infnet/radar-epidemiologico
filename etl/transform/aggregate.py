"""Agregação das notificações individuais do SINAN por município e semana epidemiológica."""

import polars as pl


def notificacoes_semanais(registros: pl.DataFrame, doenca: str) -> pl.DataFrame:
    """Agrega notificações individuais em contagens por município e semana epidemiológica.

    A definição de "caso provável" e o município de referência (residência ou notificação)
    dependem das conclusões do notebook de análise exploratória.
    """
    raise NotImplementedError("Implementar após a análise exploratória (notebooks/).")
