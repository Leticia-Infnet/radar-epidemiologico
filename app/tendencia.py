"""Cálculos da página de estimativa de tendência (AGENTS.md, seção 7.2), sem Streamlit.

As estimativas chegam prontas do treino no CI (`ml/train.py`): para cada doença e local,
uma linha por semana a partir da última completa, com os limites dos intervalos de 50% e
90%. Aqui elas só são recortadas, comparadas com a série observada e descritas.
"""

from dataclasses import dataclass

import pandas as pd

# Doenças com estimativa: as outras não têm volume ou frequência para o modelo.
DOENCAS = {"dengue": "Dengue", "chikungunya": "Chikungunya"}
MODELOS = {
    "ridge": "Regressão linear regularizada (Ridge)",
    "binomial_negativa": "GLM binomial negativa",
    "persistencia": "Repetir a última semana",
    "ingenuo_sazonal": "Mesma semana do ano anterior",
}
# Semanas observadas que aparecem no gráfico antes da estimativa.
SEMANAS_RECENTES = 26
# Semanas completas que formam o nível atual, com o qual a estimativa é comparada.
SEMANAS_DE_REFERENCIA = 4

ALTA, QUEDA, ESTAVEL = "Alta", "Queda", "Estável"


def estimativa_do_local(estimativas: pd.DataFrame, doenca: str, local: str) -> pd.DataFrame:
    """Linhas de um local (`BR`, UF ou município), uma por semana estimada; vazia se o
    local não tem estimativa."""
    linhas = estimativas[(estimativas["doenca"] == doenca) & (estimativas["local"] == local)]
    return linhas.sort_values("passo").reset_index(drop=True)


def serie_recente(serie: pd.DataFrame, estimativa: pd.DataFrame) -> pd.DataFrame:
    """Semanas observadas a mostrar com a estimativa: as últimas completas antes da
    primeira semana estimada (coluna `parcial` falsa) e, das semanas estimadas, as que
    já têm dado parcial publicado (`parcial` verdadeira).

    O corte é pela data, não pelo indicador de semana incompleta da série: se o ETL
    publicar dados mais novos que os do treino, as semanas estimadas continuam
    aparecendo como estimadas, com o dado observado ao lado.
    """
    inicio = pd.to_datetime(serie["inicio_semana"])
    primeira = estimativa["inicio_semana"].min()
    anteriores = serie[inicio < primeira].tail(SEMANAS_RECENTES).assign(parcial=False)
    posteriores = serie[inicio >= primeira].assign(parcial=True)
    recente = pd.concat([anteriores, posteriores], ignore_index=True)
    recente["inicio_semana"] = pd.to_datetime(recente["inicio_semana"])
    return recente[["ano_epi", "semana_epi", "inicio_semana", "casos_provaveis", "parcial"]]


@dataclass(frozen=True)
class Tendencia:
    """Comparação entre o nível atual e a última semana estimada."""

    nivel_atual: float  # média das últimas semanas completas
    estimativa: float  # última semana estimada
    inferior: float  # intervalo de 90%
    superior: float
    inicio_semana: pd.Timestamp  # início da última semana estimada
    variacao: float | None  # fração sobre o nível atual; None se o nível é zero
    direcao: str  # ALTA, QUEDA ou ESTAVEL


def tendencia(recente: pd.DataFrame, estimativa: pd.DataFrame) -> Tendencia:
    """Direção da tendência até a última semana estimada.

    Só é alta ou queda quando o intervalo de 50% inteiro fica acima ou abaixo do nível
    atual (a média das últimas `SEMANAS_DE_REFERENCIA` semanas completas). Se o
    intervalo contém o nível atual, a estimativa não distingue uma mudança: estável.
    """
    completas = recente.loc[~recente["parcial"], "casos_provaveis"]
    nivel = float(completas.tail(SEMANAS_DE_REFERENCIA).mean())
    final = estimativa.iloc[-1]
    if final["p25"] > nivel:
        direcao = ALTA
    elif final["p75"] < nivel:
        direcao = QUEDA
    else:
        direcao = ESTAVEL
    return Tendencia(
        nivel_atual=nivel,
        estimativa=float(final["estimativa"]),
        inferior=float(final["p05"]),
        superior=float(final["p95"]),
        inicio_semana=final["inicio_semana"],
        variacao=float(final["estimativa"]) / nivel - 1 if nivel > 0 else None,
        direcao=direcao,
    )


def tabela_semanal(recente: pd.DataFrame, estimativa: pd.DataFrame) -> pd.DataFrame:
    """Semanas estimadas com os intervalos e, quando existe, o dado parcial observado."""
    parcial = recente.loc[recente["parcial"], ["inicio_semana", "casos_provaveis"]]
    tabela = estimativa.merge(parcial, on="inicio_semana", how="left")
    return tabela[
        ["semana_epi", "inicio_semana", "estimativa", "p05", "p95", "p25", "p75", "casos_provaveis"]
    ].rename(columns={"casos_provaveis": "parcial"})


def desempenho(metricas: pd.DataFrame, doenca: str, local: str, modelo: str) -> pd.DataFrame:
    """Erro do modelo escolhido e dos dois baselines no backtest do local, por passo.

    MASE abaixo de 1: o erro é menor que o erro típico de repetir a mesma semana do ano
    anterior naquele local.
    """
    linhas = metricas[(metricas["doenca"] == doenca) & (metricas["local"] == local)]
    mase = linhas.pivot(index="passo", columns="modelo", values="mase")
    do_modelo = linhas[linhas["modelo"] == modelo].set_index("passo")
    return pd.DataFrame(
        {
            "passo": mase.index,
            "mae": do_modelo["mae"].reindex(mase.index).to_numpy(),
            "mase": mase[modelo].to_numpy(),
            "mase_persistencia": mase["persistencia"].to_numpy(),
            "mase_sazonal": mase["ingenuo_sazonal"].to_numpy(),
            "cobertura_90": do_modelo["cobertura_90"].reindex(mase.index).to_numpy(),
        }
    )
