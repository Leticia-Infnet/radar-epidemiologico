from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import data
from tests.test_painel import MUNICIPIOS
from tests.test_tendencia import estimativas_sinteticas, metricas_sinteticas, serie_com_parciais

PAGINA = str(Path(__file__).parents[1] / "app" / "pages" / "2_previsao.py")

SERIES = {
    ("dengue", "BR"): serie_com_parciais(2000),
    ("dengue", "26"): serie_com_parciais(100),
    ("chikungunya", "BR"): serie_com_parciais(300),
}
OLINDA, RECIFE = serie_com_parciais(40), serie_com_parciais(2)
RESUMO = pd.concat(
    serie.assign(doenca=doenca, local=local) for (doenca, local), serie in SERIES.items()
)
CASOS_PE = pd.concat([OLINDA.assign(codigo_ibge6="260960"), RECIFE.assign(codigo_ibge6="261160")])
# Brasil em alta, Pernambuco estável e Olinda em queda; Recife fica sem estimativa.
ESTIMATIVAS = pd.concat(
    [
        estimativas_sinteticas(SERIES["dengue", "BR"], "dengue", "BR", 1.5),
        estimativas_sinteticas(SERIES["dengue", "26"], "dengue", "26", 1.02),
        estimativas_sinteticas(OLINDA, "dengue", "260960", 0.5),
        estimativas_sinteticas(SERIES["chikungunya", "BR"], "chikungunya", "BR", 1.0),
    ]
)
METRICAS = pd.concat(
    [
        metricas_sinteticas("dengue", "BR"),
        metricas_sinteticas("dengue", "26"),
        metricas_sinteticas("dengue", "260960"),
        metricas_sinteticas("chikungunya", "BR"),
    ]
)
GRUPOS = {"agregado": {"modelo": "ridge"}, "municipio": {"modelo": "ridge"}}
DOENCA = {
    "corte": "2026-01-02",
    "semanas_incompletas": 4,
    "passos": 8,
    "primeiro_ano_teste": 2015,
    "grupos": GRUPOS,
}
METADADOS = {
    "treinado_em": "2026-01-05T12:00:00+00:00",
    "semanas_a_frente": 4,
    "criterio_de_municipios": {"media_minima_semanal": 5.0, "semanas": 260},
    "doencas": {"dengue": DOENCA, "chikungunya": DOENCA},
}


@pytest.fixture
def sem_rede(monkeypatch):
    """Substitui as leituras do Hub por dados sintéticos."""
    monkeypatch.setattr(data, "metadados", {"previsao": METADADOS}.get)
    monkeypatch.setattr(data, "estimativas", lambda: ESTIMATIVAS)
    monkeypatch.setattr(data, "metricas_previsao", lambda: METRICAS)
    monkeypatch.setattr(data, "municipios", lambda: MUNICIPIOS)
    monkeypatch.setattr(data, "resumo_arboviroses", lambda: RESUMO)
    monkeypatch.setattr(data, "casos_uf", lambda doenca, uf: CASOS_PE)


def _abrir() -> AppTest:
    return AppTest.from_file(PAGINA, default_timeout=30).run()


def _metricas(app):
    return {m.label: (m.value, m.delta) for m in app.metric}


def test_brasil_por_padrao(sem_rede):
    app = _abrir()
    assert not app.exception
    assert app.subheader[0].value == "Dengue · Brasil"
    numeros = _metricas(app)
    assert numeros["Nível atual (casos por semana)"][0] == "2.000"
    # A última semana estimada começa 8 semanas depois da última completa (30/11/2025).
    assert numeros["Estimativa para a semana de 25/01"] == ("3.000", "+50,0% sobre o nível atual")
    assert numeros["Intervalo de 90%"][0] == "1.800 a 4.500"
    assert numeros["Tendência"][0] == "Alta"
    assert len(app.get("plotly_chart")) == 1
    assert "Não é uma previsão oficial" in app.caption[0].value
    assert "05/01/2026" in app.caption[-1].value and "02/01/2026" in app.caption[-1].value


def test_tabelas_de_estimativas_e_de_erro(sem_rede):
    app = _abrir()
    semanal, desempenho = (d.value for d in app.dataframe)
    assert len(semanal) == 8
    assert semanal["parcial"].notna().sum() == 4
    assert desempenho["mase"].iloc[0] == pytest.approx(0.1)
    assert desempenho["mase_sazonal"].iloc[0] == pytest.approx(0.5)
    explicacao = " ".join(m.value for m in app.markdown)
    assert "regressão linear regularizada (ridge)" in explicacao
    assert "de 2015 em diante" in explicacao


def test_uf_estavel_e_municipio_em_queda(sem_rede):
    app = _abrir()
    app.selectbox[1].set_value("26").run()
    assert app.subheader[0].value == "Dengue · Pernambuco"
    assert _metricas(app)["Tendência"][0] == "Estável"

    app.selectbox[2].set_value("260960").run()
    assert not app.exception
    assert app.subheader[0].value == "Dengue · Olinda (PE)"
    numeros = _metricas(app)
    assert numeros["Nível atual (casos por semana)"][0] == "40"
    assert numeros["Estimativa para a semana de 25/01"] == ("20", "-50,0% sobre o nível atual")
    assert numeros["Tendência"][0] == "Queda"
    assert not app.info


def test_municipio_sem_volume_mostra_a_estimativa_da_uf(sem_rede):
    app = _abrir()
    app.selectbox[1].set_value("26").run()
    app.selectbox[2].set_value("261160").run()
    assert not app.exception
    assert "Recife (PE) tem poucos casos de dengue" in app.info[0].value
    assert "pelo menos 5 casos por semana nos últimos 5 anos" in app.info[0].value
    assert app.subheader[0].value == "Dengue · Pernambuco"
    assert _metricas(app)["Nível atual (casos por semana)"][0] == "100"


def test_troca_de_doenca(sem_rede):
    app = _abrir()
    assert app.selectbox[0].options == ["Dengue", "Chikungunya"]
    app.selectbox[0].set_value("chikungunya").run()
    assert app.subheader[0].value == "Chikungunya · Brasil"
    assert _metricas(app)["Tendência"][0] == "Estável"


def test_local_sem_estimativa(sem_rede):
    app = _abrir()
    app.selectbox[0].set_value("chikungunya").run()
    app.selectbox[1].set_value("26").run()
    assert not app.exception
    assert "Ainda não há estimativa de chikungunya para Pernambuco" in app.info[0].value
    assert not app.metric


def test_estimativas_ainda_nao_publicadas(sem_rede, monkeypatch):
    monkeypatch.setattr(data, "estimativas", lambda: None)
    app = _abrir()
    assert not app.exception
    assert "ainda não foram publicadas" in app.info[0].value
    assert not app.metric


def test_sem_metricas_a_pagina_segue_sem_a_tabela_de_erro(sem_rede, monkeypatch):
    monkeypatch.setattr(data, "metricas_previsao", lambda: None)
    app = _abrir()
    assert not app.exception
    assert len(app.dataframe) == 1
