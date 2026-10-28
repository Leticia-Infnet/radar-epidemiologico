"""Cliente assíncrono da API de Dados Abertos do Ministério da Saúde (DEMAS).

Comportamento verificado contra a API em 2026-10-28:

- nenhum dos endpoints usados exige autenticação;
- `limit` aceita até 1000; valores maiores são truncados para 1000 sem erro;
- `offset` desloca por registro, embora o Swagger o descreva como número da página;
- `campos` restringe as colunas, mas torna a consulta muito mais lenta no servidor
  (36 s com limit=10, e HTTP 502 por timeout do gateway com limit=100); sem ele, uma
  página de 1000 registros filtrada por município e ano leva de 1 a 2 s. Não use;
  selecione as colunas depois do download;
- os filtros de município usam o código IBGE de 6 dígitos (com 7, a resposta vem vazia);
- cada resposta é um objeto com uma única chave (`dengue`, `ubs`, ...) contendo a lista.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from common.config import Settings, get_settings

Registro = dict[str, Any]


@dataclass(frozen=True)
class Endpoint:
    """Endpoint paginado da DEMAS."""

    caminho: str
    filtros: tuple[str, ...] = ()


ENDPOINTS: dict[str, Endpoint] = {
    "dengue": Endpoint("/arboviroses/dengue", ("nu_ano", "id_municip")),
    "zika": Endpoint("/arboviroses/zikavirus", ("nu_ano", "id_municip")),
    "chikungunya": Endpoint("/arboviroses/chikungunya", ("nu_ano", "id_municip")),
    "febre_amarela": Endpoint("/arboviroses/febre-amarela-humanos-primatas-nao-humanos"),
    "ubs": Endpoint("/assistencia-a-saude/unidade-basicas-de-saude"),
}


def _deve_repetir(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class DemasClient:
    """Pagina, repete com backoff e limita a concorrência das chamadas à DEMAS."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        tentativas: int = 6,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or httpx.AsyncClient(
            base_url=self.settings.demas_base_url,
            timeout=self.settings.demas_timeout_s,
            headers={"User-Agent": self.settings.user_agent},
        )
        self._semaforo = asyncio.Semaphore(self.settings.demas_max_concurrency)
        self._tentativas = tentativas

    async def __aenter__(self) -> "DemasClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def pagina(
        self,
        caminho: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        campos: Sequence[str] | None = None,
        **filtros: str | int,
    ) -> list[Registro]:
        """Busca uma página e devolve a lista de registros."""
        params: dict[str, str | int] = {
            "limit": limit or self.settings.demas_page_size,
            "offset": offset,
            **{k: v for k, v in filtros.items() if v is not None},
        }
        if campos:
            params["campos"] = ",".join(campos)

        async for tentativa in AsyncRetrying(
            retry=retry_if_exception(_deve_repetir),
            wait=wait_exponential_jitter(initial=1, max=60),
            stop=stop_after_attempt(self._tentativas),
            reraise=True,
        ):
            with tentativa:
                async with self._semaforo:
                    resposta = await self._client.get(caminho, params=params)
                resposta.raise_for_status()

        corpo = resposta.json()
        # A chave muda por endpoint ("dengue", "ubs", ...); sempre há só uma.
        return next(iter(corpo.values())) if corpo else []

    async def todos(
        self,
        caminho: str,
        *,
        campos: Sequence[str] | None = None,
        max_registros: int | None = None,
        **filtros: str | int,
    ) -> list[Registro]:
        """Percorre todas as páginas até uma vir incompleta.

        A primeira página vai sozinha, para que consultas pequenas custem uma requisição;
        as seguintes vão em lotes concorrentes.
        """
        tamanho = self.settings.demas_page_size
        lote = self.settings.demas_max_concurrency
        registros = await self.pagina(caminho, offset=0, campos=campos, **filtros)
        completa = len(registros) == tamanho
        offset = tamanho
        while completa and not (max_registros and len(registros) >= max_registros):
            offsets = [offset + i * tamanho for i in range(lote)]
            paginas = await asyncio.gather(
                *(self.pagina(caminho, offset=o, campos=campos, **filtros) for o in offsets)
            )
            for pagina in paginas:
                registros.extend(pagina)
                if len(pagina) < tamanho:
                    completa = False
                    break
            offset += lote * tamanho
        return registros[:max_registros] if max_registros else registros

    async def contar(self, caminho: str, **filtros: str | int) -> int:
        """Conta os registros por busca exponencial e binária sobre o offset.

        A API não expõe contagem; isto custa cerca de 2·log2(N) requisições de 1 registro.
        """

        async def existe(posicao: int) -> bool:
            return bool(await self.pagina(caminho, offset=posicao, limit=1, **filtros))

        if not await existe(0):
            return 0
        baixo, alto = 0, 1
        while await existe(alto):
            baixo, alto = alto, alto * 2
        # Invariante: existe(baixo) é verdadeiro e existe(alto) é falso.
        while alto - baixo > 1:
            meio = (baixo + alto) // 2
            if await existe(meio):
                baixo = meio
            else:
                alto = meio
        return alto
