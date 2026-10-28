"""Ponto de entrada do Radar Epidemiológico (Streamlit em Hugging Face Spaces).

Com `st.navigation`, este script roda a cada interação, antes da página escolhida.
"""

import sys
from pathlib import Path

import streamlit as st

# No Space o projeto não é instalado como pacote; a raiz precisa estar no path
# para que `common`, `ml` e `rag` sejam importáveis pelas páginas.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

PAGINAS = Path(__file__).parent / "pages"


def inicio() -> None:
    st.title("Radar Epidemiológico")
    st.markdown(
        "Vigilância de **dengue, zika, chikungunya e febre amarela** a partir dos dados "
        "abertos do Ministério da Saúde."
    )
    st.markdown(
        """
- **Dashboard**: histórico de notificações por doença, ano e município.
- **Previsão**: estimativa de tendência de notificações.
- **Mapa**: unidades básicas de saúde mais próximas do seu CEP.
- **Chatbot**: orientação baseada nos manuais oficiais do Ministério da Saúde.
"""
    )
    st.caption("Esta ferramenta não substitui avaliação médica.")


st.set_page_config(page_title="Radar Epidemiológico", page_icon="🦟", layout="wide")
navegacao = st.navigation(
    [
        st.Page(inicio, title="Início", default=True),
        st.Page(PAGINAS / "1_dashboard.py", title="Dashboard"),
        st.Page(PAGINAS / "2_previsao.py", title="Previsão"),
        st.Page(PAGINAS / "3_mapa.py", title="Mapa de UBS"),
        st.Page(PAGINAS / "4_chatbot.py", title="Chatbot"),
    ]
)
navegacao.run()
