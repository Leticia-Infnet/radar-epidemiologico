"""Engenharia de atributos para a série semanal de notificações (AGENTS.md, seção 7.2)."""

import numpy as np
import pandas as pd

SEMANAS_NO_ANO = 52.18  # média de semanas epidemiológicas por ano


def termos_fourier(semana_epi: pd.Series, ordem: int = 2) -> pd.DataFrame:
    """Seno e cosseno da semana do ano, para capturar a sazonalidade anual."""
    angulo = 2 * np.pi * semana_epi.astype(float) / SEMANAS_NO_ANO
    colunas = {}
    for k in range(1, ordem + 1):
        colunas[f"sen_{k}"] = np.sin(k * angulo)
        colunas[f"cos_{k}"] = np.cos(k * angulo)
    return pd.DataFrame(colunas, index=semana_epi.index)


def defasagens(serie: pd.Series, lags: tuple[int, ...] = (1, 4, 52)) -> pd.DataFrame:
    """Valores da série em t-1, t-4 e t-52 (ou nos lags informados)."""
    return pd.DataFrame({f"lag_{k}": serie.shift(k) for k in lags})


def medias_moveis(serie: pd.Series, janelas: tuple[int, ...] = (4, 8)) -> pd.DataFrame:
    """Médias móveis que só usam o passado (deslocadas em 1 para evitar vazamento)."""
    passado = serie.shift(1)
    return pd.DataFrame({f"mm_{j}": passado.rolling(j).mean() for j in janelas})
