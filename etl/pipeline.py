"""Ponto de entrada do ETL, chamado pelo workflow `etl.yml`.

Uso: `uv run python -m etl.pipeline {municipios,ubs,arboviroses} [--sem-publicar]`

Cada etapa extrai, normaliza, valida contra o contrato Pandera e publica no repositório
*dataset* do Hub uma pasta com o Parquet e um `metadados.json`. Os arquivos só existem
num diretório temporário durante a execução.
"""

import argparse
import asyncio
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from common.config import get_settings
from etl.extract.demas import ENDPOINTS, DemasClient
from etl.extract.ibge import municipios
from etl.publish.hub import publicar_pasta
from etl.transform.schemas import UBS, Municipio
from etl.transform.ubs import normalizar_ubs

log = logging.getLogger("etl")

# Pisos de sanidade: abaixo deles a extração é tratada como falha, para que uma resposta
# vazia ou parcial da fonte nunca substitua o dataset publicado.
MINIMO_MUNICIPIOS = 5500  # 5.571 em 2026-10
MINIMO_UBS = 40_000  # 46.848 registros brutos em 2026-10


class ExtracaoIncompleta(RuntimeError):
    """A fonte devolveu menos linhas que o piso de sanidade."""


def _conferir_piso(nome: str, linhas: int, minimo: int) -> None:
    if linhas < minimo:
        raise ExtracaoIncompleta(f"{nome}: {linhas} linhas, abaixo do mínimo de {minimo}.")


def _publicar(tabela: pl.DataFrame, nome: str, metadados: dict[str, Any], publicar: bool) -> None:
    metadados = {
        "tabela": nome,
        "linhas": tabela.height,
        "extraido_em": datetime.now(UTC).isoformat(timespec="seconds"),
        **metadados,
    }
    log.info("%s: %s", nome, json.dumps(metadados, ensure_ascii=False))
    if not publicar:
        log.info("%s: validado, publicação desativada (--sem-publicar).", nome)
        return
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        tabela.write_parquet(pasta / f"{nome}.parquet")
        (pasta / "metadados.json").write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        commit = publicar_pasta(pasta, nome, f"{nome}: {tabela.height} linhas")
    log.info("%s: publicado em %s (commit %s).", nome, get_settings().hf_dataset_repo, commit)


def executar_municipios(publicar: bool = True) -> pl.DataFrame:
    """Materializa o dicionário de municípios do IBGE e publica no Hub."""
    tabela = pl.from_pandas(municipios())
    _conferir_piso("municipios", tabela.height, MINIMO_MUNICIPIOS)
    Municipio.validate(tabela)
    _publicar(tabela, "municipios", {"fonte": get_settings().ibge_municipios_url}, publicar)
    return tabela


async def _baixar_ubs() -> list[dict[str, Any]]:
    async with DemasClient() as cliente:
        return await cliente.todos(ENDPOINTS["ubs"].caminho)


def executar_ubs(publicar: bool = True) -> pl.DataFrame:
    """Extrai, normaliza, valida e publica o cadastro de UBS."""
    registros = asyncio.run(_baixar_ubs())
    _conferir_piso("ubs", len(registros), MINIMO_UBS)
    tabela, descartes = normalizar_ubs(registros)
    UBS.validate(tabela)
    metadados = {
        "fonte": get_settings().demas_base_url + ENDPOINTS["ubs"].caminho,
        "registros_brutos": len(registros),
        "descartes": descartes,
        "coordenadas_aproximadas": int(tabela["coordenada_aproximada"].sum()),
    }
    _publicar(tabela, "ubs", metadados, publicar)
    return tabela


def executar_arboviroses(publicar: bool = True) -> pl.DataFrame:
    """Extrai o recorte definido, agrega por semana epidemiológica e publica no Hub."""
    raise NotImplementedError


ETAPAS = {
    "municipios": executar_municipios,
    "ubs": executar_ubs,
    "arboviroses": executar_arboviroses,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapas", nargs="+", choices=ETAPAS)
    parser.add_argument(
        "--sem-publicar", action="store_true", help="Extrai e valida sem enviar ao Hub."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # Uma linha por requisição à DEMAS polui o log do Actions.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for etapa in args.etapas:
        ETAPAS[etapa](publicar=not args.sem_publicar)


if __name__ == "__main__":
    main()
