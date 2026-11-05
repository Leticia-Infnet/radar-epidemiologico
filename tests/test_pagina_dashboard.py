from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import data
from tests.test_painel import CASOS_PE, FEBRE_AMARELA, MUNICIPIOS, POPULACAO, RESUMO

PAGINA = str(Path(__file__).parents[1] / "app" / "pages" / "1_dashboard.py")
METADADOS = {
    "doencas": {
        "dengue": {"ano_inicial": 2024, "ultima_semana": "2025-53", "corte": "2026-01-02"},
        "zika": {"ano_inicial": 2025, "ultima_semana": "2025-53", "corte": "2026-01-02"},
    }
}


METADADOS_FEBRE_AMARELA = {
    "periodos": ["2016/2017", "2018/2019"],
    "extraido_em": "2026-11-05T01:00:00+00:00",
}


@pytest.fixture
def sem_rede(monkeypatch):
    """Substitui as leituras do Hub por dados sintéticos."""
    metadados = {"arboviroses": METADADOS, "febre_amarela": METADADOS_FEBRE_AMARELA}
    monkeypatch.setattr(data, "metadados", metadados.get)
    monkeypatch.setattr(data, "febre_amarela", lambda: FEBRE_AMARELA)
    monkeypatch.setattr(data, "municipios", lambda: MUNICIPIOS)
    monkeypatch.setattr(data, "populacao", lambda: POPULACAO)
    monkeypatch.setattr(data, "resumo_arboviroses", lambda: RESUMO)
    monkeypatch.setattr(data, "casos_uf", lambda doenca, uf: CASOS_PE)


def _abrir() -> AppTest:
    app = AppTest.from_file(PAGINA, default_timeout=30)
    return app.run()


def _metricas(app):
    return {m.label: (m.value, m.delta) for m in app.metric}


def test_brasil_por_padrao(sem_rede):
    app = _abrir()
    assert not app.exception
    assert app.subheader[0].value == "Dengue · Brasil · 2025"
    assert _metricas(app)["Casos prováveis"] == (
        "265",
        "-49,0% em relação a 2024, até a semana 53",
    )
    # População do Brasil na tabela sintética: 1.864.000.
    assert _metricas(app)["Incidência por 100 mil habitantes"][0] == "14,2"
    assert app.selectbox[2].disabled
    assert len(app.dataframe) == 1  # só a tabela semanal, sem ranking


def test_uf_mostra_ranking_e_municipio_mostra_a_serie_dele(sem_rede):
    app = _abrir()
    app.selectbox[1].set_value("26").run()
    assert app.subheader[0].value == "Dengue · Pernambuco · 2025"
    ranking = app.dataframe[0].value
    assert ranking["municipio"].tolist() == ["Olinda", "Recife"]

    app.selectbox[2].set_value("260960").run()
    assert not app.exception
    assert app.subheader[0].value == "Dengue · Olinda (PE) · 2025"
    assert _metricas(app)["Casos prováveis"][0] == "53"
    assert len(app.dataframe) == 1


def test_ano_escolhido_continua_ao_trocar_de_doenca(sem_rede):
    app = _abrir()
    app.selectbox[3].set_value(2024).run()
    app.selectbox[0].set_value("zika").run()
    assert not app.exception
    # A zika começa em 2025 na tabela sintética: 2024 não existe e o ano volta ao último.
    assert app.selectbox[3].value == 2025
    app.selectbox[0].set_value("dengue").run()
    assert app.selectbox[3].value == 2025


def test_aviso_da_zika_nos_primeiros_anos(sem_rede, monkeypatch):
    metadados = {"doencas": {**METADADOS["doencas"], "zika": {**METADADOS["doencas"]["zika"]}}}
    metadados["doencas"]["zika"]["ano_inicial"] = 2015
    monkeypatch.setattr(data, "metadados", lambda tabela: metadados)
    app = _abrir()
    app.selectbox[0].set_value("zika").run()
    app.selectbox[3].set_value(2016).run()
    assert not app.exception
    assert "notificação compulsória em 2016" in app.info[0].value


def test_febre_amarela_por_periodo_de_monitoramento(sem_rede):
    app = _abrir()
    app.selectbox[0].set_value("febre_amarela").run()
    assert not app.exception
    assert app.selectbox[3].label == "Período"
    assert app.selectbox[3].options == ["2018/2019", "2017/2018", "2016/2017"]
    app.selectbox[3].set_value("2016/2017").run()
    assert app.subheader[0].value == "Febre amarela · Brasil · 2016/2017"
    assert _metricas(app)["Casos confirmados"][0] == "10"
    assert _metricas(app)["Letalidade"][0] == "30,0%"
    assert _metricas(app)["Municípios com casos"][0] == "2"
    # Tabela de municípios e a dos meses; nomes vêm do dicionário (sintético).
    assert len(app.dataframe) == 2


def test_febre_amarela_ainda_nao_publicada(sem_rede, monkeypatch):
    def sem_tabela():
        raise FileNotFoundError("febre_amarela/febre_amarela.parquet")

    monkeypatch.setattr(data, "febre_amarela", sem_tabela)
    app = _abrir()
    app.selectbox[0].set_value("febre_amarela").run()
    assert not app.exception
    assert "ainda não foram publicados" in app.info[0].value
