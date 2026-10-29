"""Testes contra as APIs reais. Rodam só com `uv run pytest -m integracao`."""

import pytest

from etl.extract.demas import ENDPOINTS, DemasClient
from etl.transform.ubs import normalizar_ubs

pytestmark = [pytest.mark.integracao, pytest.mark.anyio]

OLINDA = "260960"


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_dengue_de_um_municipio_pequeno():
    async with DemasClient() as cliente:
        registros = await cliente.pagina(
            ENDPOINTS["dengue"].caminho, limit=50, nu_ano="2024", id_municip=OLINDA
        )
    assert registros
    assert {str(r["id_municip"]) for r in registros} == {OLINDA}
    assert {str(r["nu_ano"]) for r in registros} == {"2024"}


async def test_pagina_de_ubs_normaliza():
    async with DemasClient() as cliente:
        registros = await cliente.pagina(ENDPOINTS["ubs"].caminho, limit=200)
    tabela, descartes = normalizar_ubs(registros)
    assert tabela.height + sum(descartes.values()) == len(registros)
    assert tabela.height > 0.8 * len(registros)
