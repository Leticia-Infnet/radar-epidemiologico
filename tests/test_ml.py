import json

import numpy as np
import pandas as pd
import pytest

from ml import dados, evaluate, features, modelos, train
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


# --- Modelo de tendência: atributos, backtest, intervalos e treino -----------------------

LOCAIS = {"BR": 3000, "26": 400, "35": 900, "260960": 40, "261160": 60, "355030": 120}


def _calendario(semanas: int) -> pd.DataFrame:
    """Calendário simplificado: anos de 52 semanas a partir de 2015."""
    t = np.arange(semanas)
    return pd.DataFrame(
        {
            "ano_epi": 2015 + t // 52,
            "semana_epi": t % 52 + 1,
            "inicio_semana": pd.Timestamp("2015-01-04") + pd.to_timedelta(7 * t, unit="D"),
        }
    )


def _casos(semanas: int = 420, semente: int = 0) -> pd.DataFrame:
    """Séries sazonais com tendência de alta e ruído de Poisson."""
    rng = np.random.default_rng(semente)
    t = np.arange(semanas)
    forma = (1.2 + np.sin(2 * np.pi * t / 52)) * (1 + 2 * t / semanas)
    return pd.DataFrame({local: rng.poisson(nivel * forma) for local, nivel in LOCAIS.items()})


def test_atributos_alinhados_com_a_origem_e_o_alvo():
    casos, passo, t = _casos(), 4, 300
    tabela = features.montar_atributos(casos, _calendario(424), passo)
    linha = tabela[(tabela["local"] == "26") & (tabela["origem"] == t)].iloc[0]
    serie = casos["26"]
    assert linha["alvo_pos"] == t + passo
    assert (linha["ano_alvo"], linha["semana_alvo"]) == (2015 + 304 // 52, 304 % 52 + 1)
    assert linha["y"] == serie[t + passo]
    assert linha["base_persistencia"] == serie[t]
    assert linha["base_sazonal"] == serie[t + passo - 52]
    assert linha["lag_0"] == pytest.approx(np.log1p(serie[t]))
    assert linha["lag_3"] == pytest.approx(np.log1p(serie[t - 3]))
    assert linha["mm_4"] == pytest.approx(np.log1p(serie[t - 3 : t + 1].mean()))
    assert linha["nivel_52"] == pytest.approx(np.log1p(serie[t - 51 : t + 1].mean()))
    assert linha["saz_2"] == pytest.approx(np.log1p(serie[t + passo - 104]))
    anteriores = [serie[t + passo - 52 * k] for k in range(1, 6)]  # só há 5 anos antes
    assert linha["clima"] == pytest.approx(np.log1p(np.median(anteriores)))
    assert not tabela[features.ATRIBUTOS].isna().any().any()
    # As últimas origens ainda não têm o alvo.
    assert tabela.loc[tabela["origem"] > 419 - passo, "y"].isna().all()


def test_atributos_nao_usam_o_futuro():
    casos, passo, t = _casos(), 4, 300
    alterado = casos.copy()
    alterado.iloc[t + 1 :] = 0
    antes = features.montar_atributos(casos, _calendario(424), passo)
    depois = features.montar_atributos(alterado, _calendario(424), passo)
    ate_t = antes["origem"] <= t
    pd.testing.assert_frame_equal(
        antes.loc[ate_t, features.ATRIBUTOS], depois.loc[ate_t, features.ATRIBUTOS]
    )
    assert (depois.loc[depois["origem"] == t, "y"] == 0).all()


class _Espiao:
    """Modelo falso que registra os limites do treino e do teste de cada dobra."""

    dobras: list[tuple[int, int, int]] = []

    def ajustar(self, tabela):
        self.treino = tabela
        return self

    def prever(self, tabela):
        _Espiao.dobras.append(
            (int(self.treino["alvo_pos"].max()), int(tabela["origem"].min()), len(self.treino))
        )
        return np.zeros(len(tabela))


def test_backtest_treina_so_com_o_que_ja_era_conhecido():
    _Espiao.dobras = []
    tabela = features.montar_atributos(_casos(), _calendario(424), 4)
    previsoes = evaluate.backtest_origem_movel(tabela, {"espiao": _Espiao}, 2020)
    assert sorted(previsoes["ano_alvo"].unique()) == [2020, 2021, 2022, 2023]
    assert len(_Espiao.dobras) == 4
    for ultimo_alvo_do_treino, primeira_origem_do_teste, _ in _Espiao.dobras:
        # O alvo mais recente do treino já tinha acontecido na primeira origem do teste.
        assert ultimo_alvo_do_treino <= primeira_origem_do_teste
    tamanhos = [linhas for *_, linhas in _Espiao.dobras]
    assert tamanhos == sorted(tamanhos)  # a janela de treino só cresce
    assert previsoes["y"].notna().all()


def test_ridge_supera_os_baselines_em_serie_sazonal_com_tendencia():
    casos = _casos()
    tabela = features.montar_atributos(casos, _calendario(424), 4)
    previsoes = evaluate.backtest_origem_movel(tabela, modelos.CANDIDATOS, 2020)
    metricas = evaluate.metricas_por_local(
        previsoes, evaluate.escalas_sazonais(casos), list(modelos.CANDIDATOS)
    )
    mase_ = metricas.pivot(index="local", columns="modelo", values="mase")
    assert (mase_["ridge"] < mase_["ingenuo_sazonal"]).all()
    assert (mase_["ridge"] < mase_["persistencia"]).all()
    assert mase_["binomial_negativa"].median() < mase_["ingenuo_sazonal"].median()
    # Três anos de teste inteiros e as 4 semanas de 2023 que a série alcança.
    assert (metricas["n"] == 3 * 52 + 4).all()


def test_mase_do_backtest_usa_a_escala_da_serie_completa():
    casos = _casos()[["BR"]]
    tabela = features.montar_atributos(casos, _calendario(424), 1)
    previsoes = evaluate.backtest_origem_movel(tabela, modelos.BASELINES, 2020)
    metricas = evaluate.metricas_por_local(
        previsoes, evaluate.escalas_sazonais(casos), ["ingenuo_sazonal"]
    )
    esperado = mase(previsoes["y"], previsoes["ingenuo_sazonal"], treino=casos["BR"])
    assert metricas["mase"].iloc[0] == pytest.approx(esperado)


def test_escala_vazia_em_serie_so_de_zeros():
    escalas = evaluate.escalas_sazonais(pd.DataFrame({"a": np.zeros(120), "b": np.arange(120.0)}))
    assert np.isnan(escalas["a"])
    assert escalas["b"] == pytest.approx(52.0)


def test_intervalos_contem_a_estimativa_e_cobrem_o_nominal():
    rng = np.random.default_rng(1)
    nivel = rng.uniform(1, 8, 20_000)
    previsto = np.expm1(nivel)
    # Ruído maior nas séries pequenas, como nas contagens reais.
    real = np.expm1(nivel + rng.normal(0, 1 / nivel)).clip(min=0)
    metade = 10_000
    intervalos = evaluate.Intervalos.ajustar(real[:metade], previsto[:metade], nivel[:metade])
    limites = intervalos.aplicar(previsto[metade:], nivel[metade:])
    assert len(intervalos.limites) == evaluate.FAIXAS_DE_VOLUME - 1
    assert (limites >= 0).all().all()
    assert (limites["p05"] <= limites["p25"]).all()
    assert (limites["p25"] <= previsto[metade:]).all()
    assert (limites["p75"] >= previsto[metade:]).all()
    assert (limites["p95"] >= limites["p75"]).all()
    dentro = (real[metade:] >= limites["p05"]) & (real[metade:] <= limites["p95"])
    assert dentro.mean() == pytest.approx(0.90, abs=0.02)
    # A faixa de menor volume tem o intervalo relativo mais largo.
    largura = (np.log1p(limites["p95"]) - np.log1p(limites["p05"])).to_numpy()
    assert largura[nivel[metade:] < 2].mean() > largura[nivel[metade:] > 7].mean()


def test_intervalos_com_residuos_de_um_lado_so_ficam_colados_na_estimativa():
    real, previsto = np.full(50, 10.0), np.full(50, 20.0)  # o modelo sempre superestima
    limites = evaluate.Intervalos.ajustar(real, previsto, np.ones(50)).aplicar(
        np.array([20.0]), np.array([1.0])
    )
    assert limites["p75"].iloc[0] == 20.0 == limites["p95"].iloc[0]
    assert limites["p05"].iloc[0] < 20.0


def test_cobertura_usa_so_os_anos_de_teste_anteriores():
    casos = _casos()
    tabela = features.montar_atributos(casos, _calendario(424), 2)
    previsoes = evaluate.backtest_origem_movel(tabela, {"ridge": modelos.RidgeLog}, 2020)
    dentro = evaluate.dentro_do_intervalo(previsoes, "ridge")
    assert dentro[previsoes["ano_alvo"] == 2020].isna().all()
    assert dentro[previsoes["ano_alvo"] > 2020].isin([0.0, 1.0]).all()
    assert 0.7 < dentro.mean() <= 1.0


def test_municipios_elegiveis_pela_media_das_ultimas_semanas():
    semanas = dados.SEMANAS_DE_ELEGIBILIDADE
    painel = pd.DataFrame(
        {
            "grande": np.full(semanas + 100, 8),
            "limite": np.full(semanas + 100, dados.MEDIA_MINIMA),
            "pequeno": np.full(semanas + 100, 4),
            # Muitos casos no passado distante não contam.
            "antigo": np.r_[np.full(100, 500), np.full(semanas, 1)],
        }
    )
    assert dados.elegiveis(painel) == ["grande", "limite"]
    assert [dados.nivel_do_local(c) for c in ("BR", "26", "260960")] == ["br", "uf", "municipio"]


def test_calendario_estendido_atravessa_a_virada_do_ano():
    calendario = pd.DataFrame(
        {
            "ano_epi": [2025, 2025],
            "semana_epi": [51, 52],
            "inicio_semana": pd.to_datetime(["2025-12-14", "2025-12-21"]),
            "semana_incompleta": [False, True],
        }
    )
    estendido = dados.estender_calendario(calendario, 2)
    # 2025 tem 53 semanas epidemiológicas.
    assert estendido[["ano_epi", "semana_epi"]].to_numpy().tolist()[-2:] == [[2025, 53], [2026, 1]]
    assert estendido["inicio_semana"].iloc[-1] == pd.Timestamp("2026-01-04")
    assert len(dados.estender_calendario(calendario, 0)) == 2


def test_escolha_do_modelo_prefere_o_mais_simples_no_empate():
    assert train.escolher_modelo({"ridge": 0.50, "binomial_negativa": 0.495}) == "ridge"
    assert train.escolher_modelo({"ridge": 0.50, "binomial_negativa": 0.40}) == "binomial_negativa"
    assert train.escolher_modelo({"ridge": 0.40, "binomial_negativa": 0.50}) == "ridge"


def _series_sinteticas() -> dados.SeriesSemanais:
    casos = _casos(424)
    casos.iloc[-4:] = casos.iloc[-4:] // 3  # semanas incompletas, com dado parcial
    calendario = _calendario(424).assign(semana_incompleta=[False] * 420 + [True] * 4)
    return dados.SeriesSemanais(doenca="dengue", calendario=calendario, casos=casos)


@pytest.fixture
def calendario_simples(monkeypatch):
    """Estende o calendário sintético (anos de 52 semanas) sem o calendário real."""

    def estender(calendario, semanas):
        return _calendario(len(calendario) + semanas)

    monkeypatch.setattr(dados, "estender_calendario", estender)
    monkeypatch.setattr(train, "MINIMO_DE_SERIES", {"dengue": 6})


def test_treino_de_ponta_a_ponta(calendario_simples):
    resultado = train.treinar(_series_sinteticas())
    estimativas, metricas = resultado.estimativas, resultado.metricas

    assert len(estimativas) == len(LOCAIS) * 8
    br = estimativas[estimativas["local"] == "BR"]
    assert br["passo"].tolist() == list(range(1, 9))
    assert br["incompleta"].tolist() == [True] * 4 + [False] * 4
    # A primeira semana estimada é a que vem depois da última completa (posição 419).
    assert br["inicio_semana"].iloc[0] == _calendario(424)["inicio_semana"].iloc[420].date()
    assert set(estimativas["nivel"]) == {"br", "uf", "municipio"}
    # O dado parcial das semanas incompletas não entra: a estimativa segue o nível real.
    ultima_completa = _casos(424)["BR"].iloc[419]
    assert br["estimativa"].iloc[0] == pytest.approx(ultima_completa, rel=0.3)

    assert set(metricas["modelo"]) == set(modelos.CANDIDATOS)
    assert len(metricas) == len(LOCAIS) * 8 * len(modelos.CANDIDATOS)
    assert metricas.loc[metricas["modelo"].isin(modelos.BASELINES), "cobertura_90"].isna().all()
    assert metricas.loc[metricas["modelo"] == "ridge", "cobertura_90"].between(0, 1).all()

    resumo = resultado.resumo
    assert resumo["ultima_semana_completa"] == "2023-04"
    assert resumo["passos"] == 8 and resumo["semanas_incompletas"] == 4
    assert resumo["series"] == {"municipio": 3, "uf": 2, "br": 1}
    assert set(resumo["grupos"]) == {"agregado", "municipio"}
    for grupo in resumo["grupos"].values():
        assert grupo["mase_mediano"][grupo["modelo"]] < grupo["mase_mediano"]["ingenuo_sazonal"]
    json.dumps(resumo)  # vai para o metadados.json

    guardado = resultado.modelos["grupos"]["agregado"]
    assert sorted(guardado["passos"]) == list(range(1, 9))
    train._validar(estimativas, metricas)


def test_portao_de_qualidade_barra_modelo_pior_que_o_baseline(calendario_simples):
    resultado = train.treinar(_series_sinteticas())
    train.conferir_qualidade(resultado, 8)

    grupo = resultado.resumo["grupos"]["municipio"]
    grupo["mase_mediano"]["persistencia"] = grupo["mase_mediano"][grupo["modelo"]] / 2
    with pytest.raises(train.ModeloReprovado, match="não supera persistencia"):
        train.conferir_qualidade(resultado, 8)


def test_portao_de_qualidade_barra_estimativas_incompletas(calendario_simples, monkeypatch):
    resultado = train.treinar(_series_sinteticas())
    resultado.estimativas = resultado.estimativas.iloc[:-1]
    with pytest.raises(train.ModeloReprovado, match="faltam passos"):
        train.conferir_qualidade(resultado, 8)

    monkeypatch.setattr(train, "MINIMO_DE_SERIES", {"dengue": 600})
    with pytest.raises(train.ModeloReprovado, match="abaixo do mínimo"):
        train.conferir_qualidade(resultado, 8)
