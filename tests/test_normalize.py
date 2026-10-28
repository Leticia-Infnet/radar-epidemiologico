import math

import pytest

from etl.transform.normalize import (
    ausente,
    codigo_cnes,
    codigo_ibge6,
    coordenada,
    dentro_do_brasil,
)


@pytest.mark.parametrize("valor", [None, "", "nan", "NaN", " null ", math.nan])
def test_ausente(valor):
    assert ausente(valor)


def test_coordenada_com_virgula_decimal():
    assert coordenada("-8,21811") == pytest.approx(-8.21811)
    assert coordenada("-35.22944") == pytest.approx(-35.22944)
    assert coordenada("nan") is None
    assert coordenada("abc") is None


def test_bounding_box():
    assert dentro_do_brasil(-8.21811, -35.22944)
    assert not dentro_do_brasil(0.0, 0.0)
    assert not dentro_do_brasil(None, -35.0)


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [("3550308", "355030"), ("355030", "355030"), (211130, "211130"), ("211130.0", "211130")],
)
def test_codigo_ibge6(entrada, esperado):
    assert codigo_ibge6(entrada) == esperado


def test_codigo_ibge6_invalido():
    assert codigo_ibge6("12345") is None
    assert codigo_ibge6("nan") is None


def test_codigo_cnes_preserva_zeros():
    assert codigo_cnes("0000302") == "0000302"
    assert codigo_cnes(302) == "0000302"
    assert codigo_cnes("12345678") is None
