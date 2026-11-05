"""Carregamento dos artefatos publicados no Hugging Face Hub, com cache do Streamlit.

O Community Cloud não tem disco persistente: tudo é lido do Hub e mantido em memória.
"""

import json
import logging
from typing import Any

import pandas as pd
import streamlit as st
from huggingface_hub import HfFileSystem
from sklearn.neighbors import BallTree

from app.geo import construir_indice
from common.config import get_settings

log = logging.getLogger(__name__)


def _url_hub(caminho: str) -> str:
    return f"hf://datasets/{get_settings().hf_dataset_repo}/{caminho}"


@st.cache_data(ttl=86400, show_spinner="Carregando dados...")
def ler_parquet(caminho: str) -> pd.DataFrame:
    """Lê um Parquet do repositório *dataset* do Hub."""
    return pd.read_parquet(_url_hub(caminho))


@st.cache_resource(ttl=86400, show_spinner="Carregando o cadastro de UBS...")
def indice_ubs() -> tuple[BallTree, pd.DataFrame]:
    """Cadastro de UBS e o índice BallTree sobre ele, compartilhados entre sessões."""
    ubs = ler_parquet("ubs/ubs.parquet")
    return construir_indice(ubs), ubs


@st.cache_resource(ttl=86400, show_spinner="Carregando as equipes das UBS...")
def profissionais_ubs() -> pd.DataFrame | None:
    """Profissionais por UBS e ocupação, indexados por CNES para consulta rápida.

    Devolve None se a tabela não estiver no Hub: o mapa funciona sem ela.
    """
    try:
        tabela = ler_parquet("profissionais/profissionais.parquet")
    except (FileNotFoundError, OSError) as exc:
        log.warning("Tabela de profissionais indisponível: %s", exc)
        return None
    return tabela.set_index("cnes").sort_index()


@st.cache_data(ttl=86400, show_spinner=False)
def metadados(tabela: str) -> dict[str, Any]:
    """`metadados.json` de uma tabela do Hub (data de extração, corte, fontes)."""
    caminho = f"datasets/{get_settings().hf_dataset_repo}/{tabela}/metadados.json"
    return json.loads(HfFileSystem().read_text(caminho, encoding="utf-8"))


@st.cache_resource(ttl=86400, show_spinner="Carregando os municípios...")
def municipios() -> pd.DataFrame:
    """Dicionário de municípios do IBGE (código de 6 dígitos, nome e UF)."""
    return ler_parquet("municipios/municipios.parquet")


@st.cache_resource(ttl=86400, show_spinner=False)
def populacao() -> pd.DataFrame:
    """População de 2022 por município, para o coeficiente de incidência."""
    return ler_parquet("populacao/populacao.parquet")


@st.cache_resource(ttl=86400, show_spinner="Carregando as séries de arboviroses...")
def resumo_arboviroses() -> pd.DataFrame:
    """Séries semanais de dengue, chikungunya e zika por UF e Brasil (cerca de 6 MB)."""
    return _tipos_compactos(ler_parquet("arboviroses/resumo.parquet"))


# A partição de dengue de SP ocupa cerca de 60 MB; poucas ficam em memória ao mesmo tempo.
@st.cache_resource(ttl=86400, max_entries=6, show_spinner="Carregando os municípios da UF...")
def casos_uf(doenca: str, uf: str) -> pd.DataFrame:
    """Séries semanais de todos os municípios de uma UF para uma doença."""
    caminho = _url_hub(f"arboviroses/doenca={doenca}/uf={uf}/casos.parquet")
    colunas = [
        "codigo_ibge6",
        "ano_epi",
        "semana_epi",
        "inicio_semana",
        "casos_provaveis",
        "obitos",
        "semana_incompleta",
    ]
    tabela = _tipos_compactos(pd.read_parquet(caminho, columns=colunas))
    tabela["codigo_ibge6"] = tabela["codigo_ibge6"].astype("category")
    return tabela


@st.cache_resource(ttl=86400, show_spinner="Carregando os casos de febre amarela...")
def febre_amarela() -> pd.DataFrame:
    """Casos confirmados de febre amarela por período, mês e município de infecção."""
    return ler_parquet("febre_amarela/febre_amarela.parquet")


def _tipos_compactos(tabela: pd.DataFrame) -> pd.DataFrame:
    """Inteiros de 32 bits e datas como datetime64, para ocupar menos memória."""
    return tabela.astype(
        {
            "ano_epi": "int32",
            "semana_epi": "int32",
            "casos_provaveis": "int32",
            "obitos": "int32",
            "inicio_semana": "datetime64[s]",
        }
    )
