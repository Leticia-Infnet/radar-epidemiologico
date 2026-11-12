from datetime import date

import pandera.errors
import polars as pl
import pytest

from etl.transform.schemas import UBS, EstimativaSemanal, MetricaBacktest

UBS_VALIDA = {
    "cnes": ["0000302"],
    "nome": ["USF SANTO ESTEVAO"],
    "codigo_ibge6": ["260290"],
    "uf_codigo": ["26"],
    "logradouro": ["RUA DO CEMITERIO"],
    "bairro": [None],
    "latitude": [-8.21811],
    "longitude": [-35.22944],
    "coordenada_aproximada": [False],
}


def test_ubs_valida():
    UBS.validate(pl.DataFrame(UBS_VALIDA, schema_overrides={"bairro": pl.String}))


def test_ubs_fora_do_brasil_e_rejeitada():
    invalida = pl.DataFrame(
        {**UBS_VALIDA, "latitude": [40.0]}, schema_overrides={"bairro": pl.String}
    )
    with pytest.raises(pandera.errors.SchemaError):
        UBS.validate(invalida)


ESTIMATIVA_VALIDA = {
    "doenca": ["dengue"],
    "nivel": ["municipio"],
    "local": ["260960"],
    "ano_epi": [2026],
    "semana_epi": [42],
    "inicio_semana": [date(2026, 10, 18)],
    "passo": [1],
    "incompleta": [True],
    "estimativa": [40.0],
    "p05": [22.0],
    "p25": [33.0],
    "p75": [48.5],
    "p95": [70.0],
}


def test_estimativa_valida():
    EstimativaSemanal.validate(pl.DataFrame(ESTIMATIVA_VALIDA))


@pytest.mark.parametrize(
    "alteracao",
    [
        {"p25": [41.0]},
        {"p95": [45.0]},
        {"p05": [-1.0]},
        {"local": ["2609600"]},
        {"doenca": ["zika"]},
    ],
)
def test_estimativa_invalida_e_rejeitada(alteracao):
    with pytest.raises(pandera.errors.SchemaError):
        EstimativaSemanal.validate(pl.DataFrame({**ESTIMATIVA_VALIDA, **alteracao}))


def test_metrica_do_backtest_aceita_baseline_sem_cobertura():
    metrica = {
        "doenca": ["dengue", "dengue"],
        "nivel": ["br", "br"],
        "local": ["BR", "BR"],
        "passo": [1, 1],
        "modelo": ["ridge", "persistencia"],
        "mae": [900.0, 1200.0],
        "mase": [0.12, 0.17],
        "cobertura_90": [0.93, None],
        "n": [600, 600],
    }
    MetricaBacktest.validate(pl.DataFrame(metrica))
    with pytest.raises(pandera.errors.SchemaError):
        MetricaBacktest.validate(pl.DataFrame({**metrica, "modelo": ["ridge", "ridge"]}))
