"""Carregamento dos artefatos publicados no Hugging Face Hub, com cache do Streamlit.

O Space não tem disco persistente: tudo é lido do Hub no startup e mantido em memória.
"""

import pandas as pd
import streamlit as st

from common.config import get_settings


def _url_hub(caminho: str) -> str:
    return f"hf://datasets/{get_settings().hf_dataset_repo}/{caminho}"


@st.cache_data(ttl=86400, show_spinner="Carregando dados...")
def ler_parquet(caminho: str) -> pd.DataFrame:
    """Lê um Parquet do repositório *dataset* do Hub."""
    return pd.read_parquet(_url_hub(caminho))
