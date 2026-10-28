import numpy as np
import pandas as pd
import pytest

from ml.evaluate import baseline_sazonal, mae, mase
from ml.features import defasagens, medias_moveis, termos_fourier


def test_termos_fourier_limitados():
    fourier = termos_fourier(pd.Series(range(1, 53)), ordem=2)
    assert list(fourier.columns) == ["sen_1", "cos_1", "sen_2", "cos_2"]
    assert fourier.abs().max().max() <= 1


def test_medias_moveis_sem_vazamento():
    serie = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mm = medias_moveis(serie, janelas=(2,))
    # A média em t usa só t-1 e t-2.
    assert mm["mm_2"].iloc[2] == pytest.approx(1.5)


def test_defasagens():
    lags = defasagens(pd.Series([10, 20, 30]), lags=(1,))
    assert np.isnan(lags["lag_1"].iloc[0])
    assert lags["lag_1"].iloc[2] == 20


def test_baseline_sazonal_tem_mase_um_no_treino():
    semanas = np.arange(156)
    serie = pd.Series(100 + 50 * np.sin(2 * np.pi * semanas / 52) + semanas)
    previsto = baseline_sazonal(serie)
    assert mase(serie, previsto, treino=serie) == pytest.approx(1.0)
    assert mae(serie, previsto) == pytest.approx(52.0)
