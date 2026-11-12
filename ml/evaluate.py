"""Baseline, métricas, backtest de origem móvel e intervalos da estimativa."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.modelos import Modelo

# Quantis dos resíduos que viram os intervalos de 50% (p25 a p75) e 90% (p05 a p95).
QUANTIS = {"p05": 0.05, "p25": 0.25, "p75": 0.75, "p95": 0.95}
# Os resíduos em escala log são bem maiores nas séries pequenas: os quantis são
# calculados por faixa de volume (o atributo `nivel_52`).
FAIXAS_DE_VOLUME = 5
MINIMO_POR_FAIXA = 200


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


def backtest_origem_movel(
    tabela: pd.DataFrame,
    candidatos: Mapping[str, Callable[[], Modelo]],
    primeiro_ano_teste: int,
) -> pd.DataFrame:
    """Estimativas fora da amostra de cada candidato, um ano de teste por vez.

    Para cada ano a partir de `primeiro_ano_teste`, os modelos são reajustados só com o
    que já se conhecia na primeira origem daquele ano: as linhas cuja semana-alvo veio
    antes dela. Isso tira do treino as últimas `passo - 1` semanas anteriores ao ano de
    teste, que ainda não tinham acontecido. A janela de treino cresce a cada ano.
    """
    conhecidas = tabela[tabela["y"].notna()]
    passo = int(tabela["passo"].iloc[0])
    partes = []
    for ano in sorted(a for a in conhecidas["ano_alvo"].unique() if a >= primeiro_ano_teste):
        teste = conhecidas[conhecidas["ano_alvo"] == ano]
        treino = conhecidas[conhecidas["alvo_pos"] <= teste["alvo_pos"].min() - passo]
        if treino.empty:
            raise ValueError(f"Sem treino antes de {ano}: escolha um ano de teste posterior.")
        previsoes = teste[["local", "passo", "alvo_pos", "ano_alvo", "y", "nivel_52"]].copy()
        for nome, criar in candidatos.items():
            previsoes[nome] = criar().ajustar(treino).prever(teste)
        partes.append(previsoes)
    return pd.concat(partes, ignore_index=True)


def escalas_sazonais(casos: pd.DataFrame, periodo: int = 52) -> pd.Series:
    """Escala do MASE de cada local: o erro absoluto médio do baseline sazonal em toda a
    série de semanas completas.

    A escala usa a série inteira, e não só o que vinha antes de cada ano de teste,
    porque muitos municípios quase não tinham casos nos primeiros anos (a chikungunya
    chegou a eles depois): a escala desses anos seria perto de zero e o MASE, enorme para
    qualquer modelo. Série sem nenhum erro (só zeros, por exemplo) fica sem escala.
    """
    escalas = (casos - casos.shift(periodo)).abs().mean()
    return escalas.where(escalas > 0)


def metricas_por_local(
    previsoes: pd.DataFrame, escalas: pd.Series, modelos: list[str]
) -> pd.DataFrame:
    """MAE, MASE e número de semanas avaliadas por local e modelo.

    MASE abaixo de 1 indica que o modelo erra menos que o baseline sazonal costuma errar
    naquele local.
    """
    local = previsoes["local"].astype(str)
    escala = local.map(escalas).to_numpy()
    partes = []
    for modelo in modelos:
        erro = (previsoes["y"] - previsoes[modelo]).abs()
        por_local = pd.DataFrame({"local": local, "erro": erro, "escalado": erro / escala}).groupby(
            "local"
        )
        partes.append(
            pd.DataFrame(
                {
                    "modelo": modelo,
                    "mae": por_local["erro"].mean(),
                    "mase": por_local["escalado"].mean(),
                    "n": por_local["erro"].size(),
                }
            ).reset_index()
        )
    return pd.concat(partes, ignore_index=True)


@dataclass(frozen=True)
class Intervalos:
    """Quantis empíricos dos resíduos fora da amostra, em escala `log1p`, por faixa de
    volume da série."""

    limites: np.ndarray  # fronteiras entre as faixas de `nivel_52`
    quantis: np.ndarray  # uma linha por faixa, uma coluna por quantil de `QUANTIS`

    @classmethod
    def ajustar(cls, real: np.ndarray, previsto: np.ndarray, nivel: np.ndarray) -> "Intervalos":
        """Calcula os quantis a partir de estimativas que o modelo fez sem ver o real."""
        residuo = np.log1p(real) - np.log1p(previsto)
        faixas = min(FAIXAS_DE_VOLUME, max(len(residuo) // MINIMO_POR_FAIXA, 1))
        limites = np.unique(np.quantile(nivel, np.linspace(0, 1, faixas + 1)[1:-1]))
        faixa = np.searchsorted(limites, nivel, side="right")
        niveis = list(QUANTIS.values())
        geral = np.quantile(residuo, niveis)
        quantis = np.vstack(
            [
                np.quantile(residuo[faixa == i], niveis) if (faixa == i).any() else geral
                for i in range(len(limites) + 1)
            ]
        )
        return cls(limites=limites, quantis=quantis)

    def aplicar(self, previsto: np.ndarray, nivel: np.ndarray) -> pd.DataFrame:
        """Limites dos intervalos em torno de cada estimativa, em casos.

        Os limites nunca são negativos e sempre contêm a estimativa: se os resíduos de
        uma faixa forem todos de um lado, o limite daquele lado é a própria estimativa.
        """
        faixa = np.searchsorted(self.limites, nivel, side="right")
        limites = np.expm1(np.log1p(previsto)[:, None] + self.quantis[faixa]).clip(min=0)
        tabela = pd.DataFrame(limites, columns=list(QUANTIS))
        tabela["p25"] = np.minimum(tabela["p25"], previsto)
        tabela["p75"] = np.maximum(tabela["p75"], previsto)
        tabela["p05"] = np.minimum(tabela["p05"], tabela["p25"])
        tabela["p95"] = np.maximum(tabela["p95"], tabela["p75"])
        return tabela


def dentro_do_intervalo(previsoes: pd.DataFrame, modelo: str) -> pd.Series:
    """Para cada estimativa do backtest, se o real caiu no intervalo de 90%.

    Os quantis de cada ano de teste vêm só dos resíduos dos anos de teste anteriores,
    para o intervalo não ser avaliado nos dados que o calibraram. O primeiro ano de
    teste fica sem resposta.
    """
    dentro = pd.Series(np.nan, index=previsoes.index)
    anos = sorted(previsoes["ano_alvo"].unique())
    for ano in anos[1:]:
        antes, atual = previsoes[previsoes["ano_alvo"] < ano], previsoes["ano_alvo"] == ano
        intervalos = Intervalos.ajustar(
            antes["y"].to_numpy(), antes[modelo].to_numpy(), antes["nivel_52"].to_numpy()
        )
        limites = intervalos.aplicar(
            previsoes.loc[atual, modelo].to_numpy(), previsoes.loc[atual, "nivel_52"].to_numpy()
        )
        real = previsoes.loc[atual, "y"].to_numpy()
        dentro[atual] = ((real >= limites["p05"]) & (real <= limites["p95"])).to_numpy(float)
    return dentro
