import pandas as pd
import pytest

from app import graficos, tendencia
from tests.test_painel import serie_sintetica

PASSOS = 8
INCOMPLETAS = 4


def serie_com_parciais(casos_por_semana: int) -> pd.DataFrame:
    """Dois anos com `casos_por_semana`; as 4 últimas semanas incompletas, pela metade."""
    serie = serie_sintetica({2024: casos_por_semana, 2025: casos_por_semana}, INCOMPLETAS)
    serie.loc[serie["semana_incompleta"], "casos_provaveis"] = casos_por_semana // 2
    return serie


def estimativas_sinteticas(
    serie: pd.DataFrame, doenca: str, local: str, fator_final: float
) -> pd.DataFrame:
    """Estimativa que vai do nível das semanas completas até `fator_final` vezes ele, nas
    4 semanas incompletas e nas 4 seguintes."""
    completas = serie[~serie["semana_incompleta"]]
    nivel = float(completas["casos_provaveis"].iloc[-1])
    inicio = pd.Timestamp(completas["inicio_semana"].iloc[-1])
    passos = pd.Series(range(1, PASSOS + 1))
    centro = nivel * (1 + (fator_final - 1) * passos / PASSOS)
    return pd.DataFrame(
        {
            "doenca": doenca,
            "nivel": {2: "br", 6: "municipio"}.get(len(local), "uf") if local != "BR" else "br",
            "local": local,
            "ano_epi": 2025,
            "semana_epi": 49 + passos,
            "inicio_semana": [inicio + pd.Timedelta(weeks=int(k)) for k in passos],
            "passo": passos,
            "incompleta": passos <= INCOMPLETAS,
            "estimativa": centro,
            "p05": centro * 0.6,
            "p25": centro * 0.9,
            "p75": centro * 1.1,
            "p95": centro * 1.5,
        }
    )


SERIE = serie_com_parciais(20)
ALTA = estimativas_sinteticas(SERIE, "dengue", "BR", 1.5)


def test_estimativa_do_local():
    outras = estimativas_sinteticas(SERIE, "chikungunya", "BR", 1.0)
    tabela = pd.concat([outras, ALTA.iloc[::-1]])
    do_local = tendencia.estimativa_do_local(tabela, "dengue", "BR")
    assert do_local["passo"].tolist() == list(range(1, PASSOS + 1))
    assert tendencia.estimativa_do_local(tabela, "dengue", "260960").empty


def test_serie_recente_separa_completas_de_parciais():
    recente = tendencia.serie_recente(SERIE, ALTA)
    assert len(recente) == tendencia.SEMANAS_RECENTES + INCOMPLETAS
    assert recente["parcial"].sum() == INCOMPLETAS
    assert (recente.loc[~recente["parcial"], "casos_provaveis"] == 20).all()
    assert (recente.loc[recente["parcial"], "casos_provaveis"] == 10).all()
    # A última semana completa vem logo antes da primeira estimada.
    ultima = recente.loc[~recente["parcial"], "inicio_semana"].max()
    assert ALTA["inicio_semana"].min() - ultima == pd.Timedelta(weeks=1)


def test_serie_recente_segue_a_data_do_treino_se_o_dado_for_mais_novo():
    """O ETL publicou mais uma semana depois do treino: ela aparece ao lado da estimativa."""
    mais_nova = SERIE.assign(semana_incompleta=False)
    recente = tendencia.serie_recente(mais_nova, ALTA)
    assert recente["parcial"].sum() == INCOMPLETAS


@pytest.mark.parametrize(
    ("fator", "direcao", "variacao"),
    [
        (1.5, tendencia.ALTA, 0.5),
        (0.5, tendencia.QUEDA, -0.5),
        (1.05, tendencia.ESTAVEL, 0.05),  # o intervalo de 50% (18,9 a 23,1) contém 20
    ],
)
def test_tendencia(fator, direcao, variacao):
    estimativa = estimativas_sinteticas(SERIE, "dengue", "BR", fator)
    numeros = tendencia.tendencia(tendencia.serie_recente(SERIE, estimativa), estimativa)
    assert numeros.nivel_atual == 20  # as semanas parciais não entram no nível atual
    assert numeros.direcao == direcao
    assert numeros.variacao == pytest.approx(variacao)
    assert numeros.estimativa == pytest.approx(20 * fator)
    assert (numeros.inferior, numeros.superior) == pytest.approx((12 * fator, 30 * fator))
    assert numeros.inicio_semana == estimativa["inicio_semana"].iloc[-1]


def test_tendencia_sem_casos_recentes_nao_tem_variacao():
    serie = serie_com_parciais(0)
    estimativa = estimativas_sinteticas(serie, "dengue", "BR", 1.0)
    numeros = tendencia.tendencia(tendencia.serie_recente(serie, estimativa), estimativa)
    assert numeros.variacao is None
    assert numeros.direcao == tendencia.ESTAVEL


def test_tabela_semanal_traz_o_dado_parcial_so_nas_semanas_incompletas():
    tabela = tendencia.tabela_semanal(tendencia.serie_recente(SERIE, ALTA), ALTA)
    assert len(tabela) == PASSOS
    assert tabela["parcial"].tolist()[:INCOMPLETAS] == [10] * INCOMPLETAS
    assert tabela["parcial"].iloc[INCOMPLETAS:].isna().all()


def metricas_sinteticas(doenca: str, local: str) -> pd.DataFrame:
    linhas = [
        {
            "doenca": doenca,
            "local": local,
            "passo": passo,
            "modelo": modelo,
            "mae": 10.0 * passo * fator,
            "mase": 0.1 * passo * fator,
            "cobertura_90": 0.9 if modelo == "ridge" else None,
            "n": 500,
        }
        for passo in range(1, PASSOS + 1)
        for modelo, fator in (("ridge", 1), ("persistencia", 2), ("ingenuo_sazonal", 5))
    ]
    return pd.DataFrame(linhas)


def test_desempenho_compara_o_modelo_com_os_baselines():
    metricas = pd.concat([metricas_sinteticas("dengue", "BR"), metricas_sinteticas("dengue", "26")])
    tabela = tendencia.desempenho(metricas, "dengue", "BR", "ridge")
    assert tabela["passo"].tolist() == list(range(1, PASSOS + 1))
    assert tabela["mase"].iloc[3] == pytest.approx(0.4)
    assert tabela["mase_persistencia"].iloc[3] == pytest.approx(0.8)
    assert tabela["mase_sazonal"].iloc[3] == pytest.approx(2.0)
    assert tabela["mae"].iloc[3] == pytest.approx(40.0)
    assert (tabela["cobertura_90"] == 0.9).all()


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_grafico_de_tendencia(tema):
    recente = tendencia.serie_recente(SERIE, ALTA)
    figura = graficos.grafico_tendencia(recente, ALTA, tema=tema)
    tracos = {t.name: t for t in figura.data if t.name}
    assert set(tracos) == {
        "Intervalo de 90%",
        "Intervalo de 50%",
        "Observado",
        "Estimativa",
        "Dado parcial",
    }
    cores = graficos.paleta(tema)
    assert tracos["Observado"].line.color == cores.foco
    assert tracos["Observado"].line.dash is None
    assert tracos["Estimativa"].line.dash == "dash"
    assert tracos["Dado parcial"].line.color == cores.referencia
    assert len(tracos["Observado"].x) == tendencia.SEMANAS_RECENTES
    assert len(tracos["Estimativa"].x) == PASSOS
    # As faixas partem do último ponto observado, sem largura.
    faixa = tracos["Intervalo de 90%"]
    assert len(faixa.x) == PASSOS + 1
    assert faixa.y[0] == 20
    assert faixa.y[-1] == pytest.approx(ALTA["p05"].iloc[-1])
    # Sombra só nas semanas incompletas.
    (sombra,) = figura.layout.shapes
    assert pd.Timestamp(sombra.x1) - pd.Timestamp(sombra.x0) == pd.Timedelta(weeks=INCOMPLETAS)


def test_grafico_de_tendencia_sem_dado_parcial():
    recente = tendencia.serie_recente(SERIE, ALTA)
    figura = graficos.grafico_tendencia(recente[~recente["parcial"]], ALTA)
    assert "Dado parcial" not in {t.name for t in figura.data}
