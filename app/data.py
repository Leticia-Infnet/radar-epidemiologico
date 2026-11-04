"""Carregamento dos artefatos publicados no Hugging Face Hub, com cache do Streamlit.

O Community Cloud não tem disco persistente: tudo é lido do Hub e mantido em memória.
"""

import logging

import pandas as pd
import streamlit as st
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
