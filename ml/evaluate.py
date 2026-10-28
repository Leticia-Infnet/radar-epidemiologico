"""Baseline e métricas para o backtest de origem móvel."""

import numpy as np
import pandas as pd


def baseline_sazonal(serie: pd.Series, periodo: int = 52) -> pd.Series:
    """Previsão ingênua sazonal: o valor da mesma semana do ano anterior."""
    return serie.shift(periodo)


def mae(real: pd.Series, previsto: pd.Series) -> float:
    """Erro absoluto médio, ignorando posições sem previsão."""
    erro = (real - previsto).abs().dropna()
    return float(erro.mean())


def mase(real: pd.Series, previsto: pd.Series, treino: pd.Series, periodo: int = 52) -> float:
    """MASE com escala do erro do baseline sazonal no treino.

    Valor abaixo de 1 indica que o modelo supera o baseline sazonal.
    """
    escala = (treino - treino.shift(periodo)).abs().dropna().mean()
    if not escala or np.isnan(escala):
        return float("nan")
    return mae(real, previsto) / float(escala)
