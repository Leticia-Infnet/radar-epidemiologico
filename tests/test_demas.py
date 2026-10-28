import httpx
import pytest

from common.config import Settings
from etl.extract.demas import DemasClient

TOTAL = 2350


def _transporte(falhas_iniciais: int = 0) -> httpx.MockTransport:
    """Simula a DEMAS: offset por registro, teto de página e falhas transitórias."""
    estado = {"falhas": falhas_iniciais}

    def responder(request: httpx.Request) -> httpx.Response:
        if estado["falhas"]:
            estado["falhas"] -= 1
            return httpx.Response(503)
        limit = min(int(request.url.params["limit"]), 1000)
        offset = int(request.url.params["offset"])
        registros = [{"id": i} for i in range(offset, min(offset + limit, TOTAL))]
        return httpx.Response(200, json={"dengue": registros})

    return httpx.MockTransport(responder)


def _cliente(transporte: httpx.MockTransport) -> DemasClient:
    settings = Settings(demas_page_size=1000, demas_max_concurrency=2)
    http = httpx.AsyncClient(base_url="https://demas.teste", transport=transporte)
    return DemasClient(settings, client=http)


@pytest.mark.anyio
async def test_todos_percorre_todas_as_paginas():
    async with _cliente(_transporte()) as cliente:
        registros = await cliente.todos("/arboviroses/dengue")
    assert [r["id"] for r in registros] == list(range(TOTAL))


@pytest.mark.anyio
async def test_todos_respeita_max_registros():
    async with _cliente(_transporte()) as cliente:
        registros = await cliente.todos("/arboviroses/dengue", max_registros=1500)
    assert len(registros) == 1500


@pytest.mark.anyio
async def test_contar_por_busca_binaria():
    async with _cliente(_transporte()) as cliente:
        assert await cliente.contar("/arboviroses/dengue") == TOTAL


@pytest.mark.anyio
async def test_repete_apos_erro_transitorio(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _sem_espera)
    async with _cliente(_transporte(falhas_iniciais=2)) as cliente:
        pagina = await cliente.pagina("/arboviroses/dengue", limit=10)
    assert len(pagina) == 10


async def _sem_espera(*_args, **_kwargs):
    return None


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_cliente_padrao_monta_cabecalhos():
    DemasClient(Settings())


@pytest.mark.anyio
async def test_consulta_pequena_custa_uma_requisicao():
    chamadas = []

    def responder(request: httpx.Request) -> httpx.Response:
        chamadas.append(request.url.params["offset"])
        return httpx.Response(200, json={"dengue": [{"id": 1}]})

    async with _cliente(httpx.MockTransport(responder)) as cliente:
        assert len(await cliente.todos("/arboviroses/dengue")) == 1
    assert chamadas == ["0"]
