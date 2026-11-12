"""Modelos candidatos à estimativa de tendência (AGENTS.md, seção 7.2).

Todos recebem a tabela de `ml.features.montar_atributos` e devolvem casos estimados para
a semana-alvo. A progressão vai dos baselines (os números a bater) à regressão linear
com atributos e ao GLM de contagem. Cada modelo é global: um ajuste por passo com as
séries de vários locais juntas (o treino separa Brasil e UFs dos municípios).
"""

from typing import Protocol, Self

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import ATRIBUTOS


class Modelo(Protocol):
    """Interface comum: ajusta com as linhas de treino e estima as de teste."""

    def ajustar(self, tabela: pd.DataFrame) -> Self: ...

    def prever(self, tabela: pd.DataFrame) -> np.ndarray: ...


class IngenuoSazonal:
    """Repete a mesma semana do ano anterior."""

    def ajustar(self, tabela: pd.DataFrame) -> Self:
        return self

    def prever(self, tabela: pd.DataFrame) -> np.ndarray:
        return tabela["base_sazonal"].to_numpy(dtype=float)


class Persistencia:
    """Repete a última semana conhecida. Difícil de bater nos passos curtos."""

    def ajustar(self, tabela: pd.DataFrame) -> Self:
        return self

    def prever(self, tabela: pd.DataFrame) -> np.ndarray:
        return tabela["base_persistencia"].to_numpy(dtype=float)


# Dispersão suposta ao ponderar a regressão em escala log (ver `RidgeLog`).
DISPERSAO = 0.05


class RidgeLog:
    """Regressão linear regularizada sobre `log1p` dos casos, ponderada pelo volume.

    O log de uma contagem pequena oscila muito mais que o de uma grande: a variância é
    perto de 1/média + dispersão. Cada linha pesa o inverso disso, com a média das 4
    últimas semanas no lugar da média. Sem o peso, as semanas de poucos casos dominam o
    ajuste e o modelo perde da persistência justo nas semanas de muitos casos.
    """

    def __init__(self, alfa: float = 1.0) -> None:
        self.alfa = alfa
        self.regressor = make_pipeline(StandardScaler(), Ridge(alpha=alfa))

    def ajustar(self, tabela: pd.DataFrame) -> Self:
        volume = 1 + np.expm1(tabela["mm_4"].to_numpy())
        self.regressor.fit(
            tabela[ATRIBUTOS].to_numpy(),
            np.log1p(tabela["y"].to_numpy()),
            ridge__sample_weight=volume / (1 + DISPERSAO * volume),
        )
        return self

    def prever(self, tabela: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.regressor.predict(tabela[ATRIBUTOS].to_numpy())).clip(min=0)


class BinomialNegativa:
    """GLM binomial negativa com link log: a verossimilhança certa para contagens com
    sobredispersão. Guarda só os coeficientes, não os dados do ajuste."""

    # Teto do preditor linear: exp(25) passa de 70 bilhões de casos.
    TETO = 25.0
    # O ajuste do statsmodels guarda várias cópias da matriz (4,5 GB com 1 milhão de
    # linhas). Para 16 coeficientes, uma amostra deste tamanho basta.
    MAXIMO_DE_LINHAS = 250_000

    def __init__(self) -> None:
        self.alfa = float("nan")
        self.coeficientes = np.empty(0)

    @staticmethod
    def _matriz(tabela: pd.DataFrame) -> np.ndarray:
        return sm.add_constant(tabela[ATRIBUTOS].to_numpy(), has_constant="add")

    def ajustar(self, tabela: pd.DataFrame) -> Self:
        if len(tabela) > self.MAXIMO_DE_LINHAS:
            tabela = tabela.sample(self.MAXIMO_DE_LINHAS, random_state=0)
        x, y = self._matriz(tabela), tabela["y"].to_numpy(dtype=float)
        poisson = sm.GLM(y, x, family=sm.families.Poisson()).fit()
        media = poisson.mu
        # Dispersão pela regressão auxiliar de Cameron e Trivedi: Var = média + alfa·média².
        self.alfa = max(float(((y - media) ** 2 - y).sum() / (media**2).sum()), 1e-4)
        ajuste = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=self.alfa)).fit(
            start_params=poisson.params
        )
        self.coeficientes = np.asarray(ajuste.params)
        return self

    def prever(self, tabela: pd.DataFrame) -> np.ndarray:
        return np.exp((self._matriz(tabela) @ self.coeficientes).clip(max=self.TETO))


BASELINES = {"ingenuo_sazonal": IngenuoSazonal, "persistencia": Persistencia}
# Do mais simples ao mais complexo: em empate, fica o primeiro.
MODELOS = {"ridge": RidgeLog, "binomial_negativa": BinomialNegativa}
CANDIDATOS = BASELINES | MODELOS
