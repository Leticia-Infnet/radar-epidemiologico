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


# Atributos do modelo de tendência, todos conhecidos na semana de origem.
ATRIBUTOS = [
    "lag_0",
    "lag_1",
    "lag_2",
    "lag_3",
    "mm_4",
    "mm_8",
    "nivel_52",
    "saz_1",
    "saz_2",
    "cresc_saz",
    "clima",
    "sen_1",
    "cos_1",
    "sen_2",
    "cos_2",
]
# Semanas de história antes da primeira origem: o atributo `saz_2` olha dois anos atrás.
HISTORICO_MINIMO = 104
ANOS_DE_CLIMATOLOGIA = 10
PERIODO = 52


def _deslocar(valores: np.ndarray, k: int) -> np.ndarray:
    """Linha `t` passa a ter o valor de `t - k` (`k` negativo olha para a frente)."""
    saida = np.full_like(valores, np.nan, dtype=float)
    if k == 0:
        saida[:] = valores
    elif 0 < k < len(valores):
        saida[k:] = valores[:-k]
    elif -len(valores) < k < 0:
        saida[:k] = valores[-k:]
    return saida


def _media_movel(valores: np.ndarray, janela: int) -> np.ndarray:
    """Média das últimas `janela` semanas, incluindo a da própria linha."""
    acumulado = np.cumsum(np.vstack([np.zeros((1, valores.shape[1])), valores]), axis=0)
    saida = np.full_like(valores, np.nan, dtype=float)
    saida[janela - 1 :] = (acumulado[janela:] - acumulado[:-janela]) / janela
    return saida


def montar_atributos(casos: pd.DataFrame, calendario: pd.DataFrame, passo: int) -> pd.DataFrame:
    """Uma linha por local e semana de origem `t`, para estimar a semana `t + passo`.

    `casos` tem as semanas completas nas linhas e os locais nas colunas; `calendario`
    cobre essas semanas e as `passo` seguintes. Os atributos (`ATRIBUTOS`) usam só o que
    se conhece até `t`, na escala `log1p`:

    - nível recente: as quatro últimas semanas e as médias de 4, 8 e 52 semanas;
    - sazonalidade da própria série: a semana-alvo um e dois anos antes (`saz_1`,
      `saz_2`), quanto a série cresceu no ano anterior entre a origem e o alvo
      (`cresc_saz`) e a mediana da semana-alvo em até 10 anos anteriores (`clima`);
    - termos de Fourier da semana-alvo.

    As colunas `y` (casos na semana-alvo; vazio se ela ainda não aconteceu),
    `base_sazonal` e `base_persistencia` acompanham os atributos. Origens sem dois anos
    de história ficam de fora.
    """
    if not 1 <= passo < PERIODO:
        raise ValueError("O passo deve ficar entre 1 e 51 semanas.")
    semanas, locais = casos.shape
    if len(calendario) < semanas + passo:
        raise ValueError("O calendário precisa cobrir as semanas-alvo.")

    y = casos.to_numpy(dtype=float)
    log = np.log1p(y)
    anteriores = np.stack(
        [_deslocar(y, PERIODO * k - passo) for k in range(1, ANOS_DE_CLIMATOLOGIA + 1)]
    )
    saz_1 = _deslocar(log, PERIODO - passo)
    matrizes = {
        "lag_0": log,
        "lag_1": _deslocar(log, 1),
        "lag_2": _deslocar(log, 2),
        "lag_3": _deslocar(log, 3),
        "mm_4": np.log1p(_media_movel(y, 4)),
        "mm_8": np.log1p(_media_movel(y, 8)),
        "nivel_52": np.log1p(_media_movel(y, PERIODO)),
        "saz_1": saz_1,
        "saz_2": _deslocar(log, 2 * PERIODO - passo),
        "cresc_saz": saz_1 - _deslocar(log, PERIODO),
        "y": _deslocar(y, -passo),
        "base_sazonal": anteriores[0],
        "base_persistencia": y,
    }

    origens = np.arange(HISTORICO_MINIMO, semanas)
    # Com dois anos de história, `anteriores` sempre tem ao menos dois valores por linha.
    matrizes["clima"] = np.full_like(y, np.nan)
    matrizes["clima"][origens] = np.log1p(np.nanmedian(anteriores[:, origens], axis=0))

    alvo = calendario.iloc[origens + passo]
    fourier = termos_fourier(alvo["semana_epi"].reset_index(drop=True))
    por_origem = {
        "origem": origens,
        "alvo_pos": origens + passo,
        "ano_alvo": alvo["ano_epi"].to_numpy(),
        "semana_alvo": alvo["semana_epi"].to_numpy(),
        **{nome: fourier[nome].to_numpy() for nome in fourier},
    }

    # Empilha local por local: a ordem das linhas é (local, origem).
    colunas = {
        "local": pd.Categorical.from_codes(
            np.repeat(np.arange(locais), len(origens)), categories=casos.columns.astype(str)
        ),
        "passo": passo,
    }
    colunas |= {nome: np.tile(valores, locais) for nome, valores in por_origem.items()}
    colunas |= {nome: m[origens].T.ravel() for nome, m in matrizes.items()}
    tabela = pd.DataFrame(colunas)
    return tabela[
        ["local", "passo", "origem", "alvo_pos", "ano_alvo", "semana_alvo"]
        + ATRIBUTOS
        + ["y", "base_sazonal", "base_persistencia"]
    ]
