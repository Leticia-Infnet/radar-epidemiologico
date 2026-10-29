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


def test_geocodifica_cep_real():
    from app.geo import geocodificar_cep

    local = geocodificar_cep("53020-140")  # Olinda (PE)
    assert -8.1 < local.latitude < -7.9
    assert -34.95 < local.longitude < -34.8


def test_detalhe_cnes_real():
    from app.geo import detalhar_estabelecimento

    detalhe = detalhar_estabelecimento("0000302")
    assert detalhe is not None
    assert detalhe["cnes"] == "0000302"


def test_rota_a_pe_real():
    from app.geo import Localizacao, distancias_a_pe, rota_a_pe

    origem = Localizacao(-8.00889, -34.85528, "teste", "")
    destino = (-8.0105, -34.8545)
    [a_pe] = distancias_a_pe(origem, [destino])
    rota = rota_a_pe(origem, *destino)
    assert a_pe is not None and rota is not None
    assert rota.distancia_m == pytest.approx(a_pe[0], rel=0.05)
    # Perfil a pé: bem mais lento que carro (cerca de 5 km/h).
    assert rota.duracao_s > rota.distancia_m / 2
