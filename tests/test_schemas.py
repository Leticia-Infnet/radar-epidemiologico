import pandera.errors
import polars as pl
import pytest

from etl.transform.schemas import UBS

UBS_VALIDA = {
    "cnes": ["0000302"],
    "nome": ["USF SANTO ESTEVAO"],
    "codigo_ibge6": ["260290"],
    "uf_codigo": ["26"],
    "logradouro": ["RUA DO CEMITERIO"],
    "bairro": [None],
    "latitude": [-8.21811],
    "longitude": [-35.22944],
}


def test_ubs_valida():
    UBS.validate(pl.DataFrame(UBS_VALIDA, schema_overrides={"bairro": pl.String}))


def test_ubs_fora_do_brasil_e_rejeitada():
    invalida = pl.DataFrame(
        {**UBS_VALIDA, "latitude": [40.0]}, schema_overrides={"bairro": pl.String}
    )
    with pytest.raises(pandera.errors.SchemaError):
        UBS.validate(invalida)
