"""Ponto de entrada do ETL, chamado pelo workflow `etl.yml`.

Uso: `uv run python -m etl.pipeline <etapas> [--sem-publicar]`, com as etapas
`municipios`, `ubs`, `profissionais`, `populacao` e `arboviroses`.

Cada etapa extrai, normaliza, valida contra o contrato Pandera e publica no repositório
*dataset* do Hub uma pasta com o(s) Parquet(s) e um `metadados.json`. Os arquivos só
existem num diretório temporário durante a execução.
"""

import argparse
import asyncio
import json
import logging
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from common.config import get_settings
from etl.extract import datasus
from etl.extract.demas import ENDPOINTS, DemasClient
from etl.extract.ibge import municipios
from etl.publish.hub import publicar_pasta
from etl.transform.aggregate import CAMPOS_SINAN, contar_por_semana, series_semanais
from etl.transform.calendario import semana_epidemiologica
from etl.transform.populacao import normalizar_populacao
from etl.transform.profissionais import CAMPOS_PF, agregar_profissionais, competencia_iso
from etl.transform.schemas import (
    UBS,
    CasosSemanais,
    Municipio,
    Populacao,
    ProfissionalUBS,
    ResumoSemanal,
)
from etl.transform.ubs import normalizar_ubs

log = logging.getLogger("etl")

# Pisos de sanidade: abaixo deles a extração é tratada como falha, para que uma resposta
# vazia ou parcial da fonte nunca substitua o dataset publicado.
MINIMO_MUNICIPIOS = 5500  # 5.571 em 2026-10
MINIMO_UBS = 40_000  # 46.848 registros brutos em 2026-10
MINIMO_UBS_COM_PROFISSIONAIS = 35_000  # em PE, 95% das UBS têm profissional (2026-09)
MINIMO_POPULACAO = 5500  # 5.570 municípios em 2026-11
# Casos prováveis somados em todo o período de cada agravo.
# Em 2026-11: 22,7 milhões de dengue (2010 a 2026), 1,65 milhão de chikungunya (2014 a
# 2026) e 342 mil de zika (2015 a 2026).
MINIMO_CASOS = {"dengue": 15_000_000, "chikungunya": 1_000_000, "zika": 200_000}

# Primeiro ano da série de cada agravo (decisão de 2026-11-04). A chikungunya entrou no
# SINAN em 2014 e a zika em 2015; a zika só passou a ser de notificação compulsória em 2016.
ANOS_INICIAIS = {"dengue": 2010, "chikungunya": 2014, "zika": 2015}


class ExtracaoIncompleta(RuntimeError):
    """A fonte devolveu menos linhas que o piso de sanidade."""


def _conferir_piso(nome: str, linhas: int, minimo: int) -> None:
    if linhas < minimo:
        raise ExtracaoIncompleta(f"{nome}: {linhas} linhas, abaixo do mínimo de {minimo}.")


def _publicar_pasta(
    pasta: Path,
    nome: str,
    linhas: int,
    metadados: dict[str, Any],
    publicar: bool,
    apagar: list[str] | None = None,
) -> None:
    """Grava o `metadados.json` na pasta já preenchida com os Parquets e envia ao Hub.

    `apagar` lista padrões de arquivos antigos da pasta no Hub que devem sair se não
    forem reenviados (por exemplo, partições que deixaram de existir).
    """
    metadados = {
        "tabela": nome,
        "linhas": linhas,
        "extraido_em": datetime.now(UTC).isoformat(timespec="seconds"),
        **metadados,
    }
    log.info("%s: %s", nome, json.dumps(metadados, ensure_ascii=False))
    if not publicar:
        log.info("%s: validado, publicação desativada (--sem-publicar).", nome)
        return
    (pasta / "metadados.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    commit = publicar_pasta(pasta, nome, f"{nome}: {linhas} linhas", apagar=apagar)
    log.info("%s: publicado em %s (commit %s).", nome, get_settings().hf_dataset_repo, commit)


def _publicar(tabela: pl.DataFrame, nome: str, metadados: dict[str, Any], publicar: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        tabela.write_parquet(pasta / f"{nome}.parquet")
        _publicar_pasta(pasta, nome, tabela.height, metadados, publicar)


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


def _cnes_das_ubs_publicadas() -> set[str]:
    url = f"hf://datasets/{get_settings().hf_dataset_repo}/ubs/ubs.parquet"
    return set(pl.from_pandas(pd.read_parquet(url, columns=["cnes"]))["cnes"])


def executar_profissionais(publicar: bool = True) -> pl.DataFrame:
    """Conta os profissionais de saúde de cada UBS a partir dos arquivos PF do CNES.

    Usa as UBS já publicadas no Hub e a competência mais recente com todas as UFs. Lê
    uma UF por vez, para que só um arquivo fique em disco temporário.
    """
    cnes_ubs = _cnes_das_ubs_publicadas()
    aamm = datasus.competencia_pf()
    ocupacoes, familias = datasus.ocupacoes_cbo()
    log.info("profissionais: competência %s, %d UBS.", aamm, len(cnes_ubs))

    def vinculos():
        for uf in datasus.UFS:
            yield from datasus.profissionais_uf(uf, aamm, CAMPOS_PF)

    tabela = agregar_profissionais(vinculos(), cnes_ubs, ocupacoes, competencia_iso(aamm), familias)
    unidades = tabela["cnes"].n_unique()
    _conferir_piso("profissionais", unidades, MINIMO_UBS_COM_PROFISSIONAIS)
    ProfissionalUBS.validate(tabela)
    metadados = {
        "fonte": f"ftp://{get_settings().datasus_ftp_host}{get_settings().datasus_cnes_pf_dir}",
        "competencia": competencia_iso(aamm),
        "ubs_com_profissionais": unidades,
        "ubs_publicadas": len(cnes_ubs),
    }
    _publicar(tabela, "profissionais", metadados, publicar)
    return tabela


async def _baixar_regioes_de_saude() -> list[dict[str, Any]]:
    async with DemasClient() as cliente:
        # O endpoint devolve no máximo 860 registros por página, qualquer que seja o limit.
        return await cliente.todos(ENDPOINTS["regioes_de_saude"].caminho, tamanho_pagina=500)


def executar_populacao(publicar: bool = True) -> pl.DataFrame:
    """População de 2022 por município, para o cálculo de incidência no dashboard."""
    registros = asyncio.run(_baixar_regioes_de_saude())
    tabela = normalizar_populacao(registros)
    _conferir_piso("populacao", tabela.height, MINIMO_POPULACAO)
    Populacao.validate(tabela)
    metadados = {
        "fonte": get_settings().demas_base_url + ENDPOINTS["regioes_de_saude"].caminho,
        "registros_brutos": len(registros),
    }
    _publicar(tabela, "populacao", metadados, publicar)
    return tabela


def _municipios_publicados() -> list[str]:
    url = f"hf://datasets/{get_settings().hf_dataset_repo}/municipios/municipios.parquet"
    return pd.read_parquet(url, columns=["codigo_ibge6"])["codigo_ibge6"].tolist()


def gerar_arboviroses(pasta: Path, municipios: list[str]) -> tuple[int, dict[str, Any]]:
    """Grava em `pasta` as séries de cada agravo e devolve o total de linhas e metadados.

    Estrutura: `doenca=<agravo>/uf=<código>/casos.parquet` por município (uma partição por
    UF, para o app ler só a UF escolhida) e `resumo.parquet` por UF e Brasil. Lê um
    arquivo do SINAN por vez e grava cada agravo antes de passar ao próximo.
    """
    linhas, resumos, metadados = 0, [], {}
    for doenca, ano_inicial in ANOS_INICIAIS.items():
        arquivos = datasus.arquivos_sinan(doenca, ano_inicial)
        contagens, publicacoes = [], {}
        descartes: dict[str, int] = {}
        for arquivo in arquivos:
            registros, publicado_em = datasus.ler_sinan(arquivo, CAMPOS_SINAN)
            contagem, motivos = contar_por_semana(registros, publicado_em.date())
            contagens.append(contagem)
            publicacoes[Path(arquivo.caminho).name] = {
                "preliminar": arquivo.preliminar,
                "publicado_em": publicado_em.date().isoformat(),
                "registros": registros.height,
            }
            for motivo, n in motivos.items():
                descartes[motivo] = descartes.get(motivo, 0) + n
            log.info("%s: %d registros, descartes %s.", arquivo.caminho, registros.height, motivos)

        corte = max(date.fromisoformat(p["publicado_em"]) for p in publicacoes.values())
        por_municipio, resumo, totais = series_semanais(
            contagens, municipios, doenca, ano_inicial, corte
        )
        if totais["casos_provaveis"] < MINIMO_CASOS[doenca]:
            raise ExtracaoIncompleta(
                f"{doenca}: {totais['casos_provaveis']} casos prováveis, abaixo do mínimo "
                f"de {MINIMO_CASOS[doenca]}."
            )
        CasosSemanais.validate(por_municipio)
        ResumoSemanal.validate(resumo)
        for (uf,), particao in por_municipio.partition_by("uf_codigo", as_dict=True).items():
            destino = pasta / f"doenca={doenca}" / f"uf={uf}"
            destino.mkdir(parents=True)
            particao.write_parquet(destino / "casos.parquet")
        resumos.append(resumo)
        linhas += por_municipio.height
        ultima = semana_epidemiologica(corte)
        metadados[doenca] = {
            "ano_inicial": ano_inicial,
            "ultima_semana": f"{ultima[0]}-{ultima[1]:02d}",
            "corte": corte.isoformat(),
            "linhas_por_municipio": por_municipio.height,
            **totais,
            "descartes": descartes,
            "arquivos": publicacoes,
        }
        del por_municipio, contagens

    pl.concat(resumos).write_parquet(pasta / "resumo.parquet")
    return linhas, metadados


def executar_arboviroses(publicar: bool = True) -> None:
    """Agrega dengue, chikungunya e zika do SINAN (FTP do DATASUS) e publica no Hub.

    Casos prováveis e óbitos por município de residência e semana de início dos
    sintomas, com zero nas semanas sem casos (decisões na seção 12 do AGENTS.md).
    """
    municipios_ = _municipios_publicados()
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        linhas, por_doenca = gerar_arboviroses(pasta, municipios_)
        s = get_settings()
        metadados = {
            "fonte": f"ftp://{s.datasus_ftp_host}{s.datasus_sinan_finais_dir} e "
            f"{s.datasus_sinan_prelim_dir}",
            "definicao": "casos prováveis (não descartados) por município de residência e "
            "semana epidemiológica de início dos sintomas",
            "doencas": por_doenca,
        }
        _publicar_pasta(pasta, "arboviroses", linhas, metadados, publicar, apagar=["*.parquet"])


ETAPAS = {
    "municipios": executar_municipios,
    "ubs": executar_ubs,
    "profissionais": executar_profissionais,
    "populacao": executar_populacao,
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
