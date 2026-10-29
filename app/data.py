"""Carregamento dos artefatos publicados no Hugging Face Hub, com cache do Streamlit.

O Community Cloud não tem disco persistente: tudo é lido do Hub e mantido em memória.
"""

import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree

from app.geo import construir_indice
from common.config import get_settings


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
