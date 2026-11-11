import httpx
import pytest

from common.config import Settings
from etl.extract.demas import DemasClient

TOTAL = 2350
DADOS = [{"id": i} for i in range(TOTAL)]

# O sentido do offset mudou em 2026-11: o cliente precisa funcionar com os dois.
SENTIDOS = pytest.mark.parametrize("por_pagina", [True, False], ids=["pagina", "registro"])


def _transporte(
    falhas_iniciais: int = 0,
    teto: int = 1000,
    por_pagina: bool = True,
    dados: list[dict] | None = None,
    chamadas: list[int] | None = None,
) -> httpx.MockTransport:
    """Simula a DEMAS: offset por página ou por registro, teto de página e falhas."""
    estado = {"falhas": falhas_iniciais}
    dados = DADOS if dados is None else dados

    def responder(request: httpx.Request) -> httpx.Response:
        if estado["falhas"]:
            estado["falhas"] -= 1
            return httpx.Response(503)
        limit = min(int(request.url.params["limit"]), teto)
        offset = int(request.url.params["offset"])
        if chamadas is not None:
            chamadas.append(offset)
        inicio = offset * limit if por_pagina else offset
        return httpx.Response(200, json={"dengue": dados[inicio : inicio + limit]})

    return httpx.MockTransport(responder)


def _cliente(transporte: httpx.MockTransport) -> DemasClient:
    settings = Settings(demas_page_size=1000, demas_max_concurrency=2)
    http = httpx.AsyncClient(base_url="https://demas.teste", transport=transporte)
    return DemasClient(settings, client=http)


@SENTIDOS
@pytest.mark.anyio
async def test_todos_percorre_todas_as_paginas(por_pagina):
    async with _cliente(_transporte(por_pagina=por_pagina)) as cliente:
        registros = await cliente.todos("/arboviroses/dengue")
    assert registros == DADOS


@SENTIDOS
@pytest.mark.anyio
async def test_todos_com_teto_de_pagina_menor_que_o_limit(por_pagina):
    # O endpoint de regiões de saúde devolve no máximo 860 registros por página.
    async with _cliente(_transporte(teto=860, por_pagina=por_pagina)) as cliente:
        incompleto = await cliente.todos("/macrorregiao-e-regiao-de-saude/municipio")
        completo = await cliente.todos(
            "/macrorregiao-e-regiao-de-saude/municipio", tamanho_pagina=500
        )
    assert len(incompleto) == 860
    assert completo == DADOS


@SENTIDOS
@pytest.mark.anyio
async def test_todos_respeita_max_registros(por_pagina):
    async with _cliente(_transporte(por_pagina=por_pagina)) as cliente:
        registros = await cliente.todos("/arboviroses/dengue", max_registros=1500)
    assert registros == DADOS[:1500]


@SENTIDOS
@pytest.mark.anyio
async def test_todos_com_linhas_identicas_no_inicio(por_pagina):
    # As duas primeiras linhas da febre amarela na DEMAS são idênticas.
    dados = [{"id": 0}, *DADOS]
    async with _cliente(_transporte(por_pagina=por_pagina, dados=dados)) as cliente:
        assert await cliente.todos("/arboviroses/febre-amarela") == dados


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("por_pagina", "offsets"),
    [(True, [0, 1, 2, 3]), (False, [0, 1, 1000, 2000])],
    ids=["pagina", "registro"],
)
async def test_todos_descobre_o_sentido_do_offset(por_pagina, offsets):
    # Com offset por página, a sondagem em offset=1 já é a segunda página; com offset
    # por registro, ela é descartada e custa uma requisição a mais.
    chamadas: list[int] = []
    async with _cliente(_transporte(por_pagina=por_pagina, chamadas=chamadas)) as cliente:
        await cliente.todos("/arboviroses/dengue")
    assert sorted(chamadas) == offsets


@SENTIDOS
@pytest.mark.anyio
async def test_contar_por_busca_binaria(por_pagina):
    async with _cliente(_transporte(por_pagina=por_pagina)) as cliente:
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
