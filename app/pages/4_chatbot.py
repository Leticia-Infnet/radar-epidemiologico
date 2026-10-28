"""Chatbot RAG sobre os manuais do Ministério da Saúde (AGENTS.md, seção 7.4)."""

import streamlit as st

st.title("Orientação sobre sintomas")
st.warning(
    "Esta ferramenta não faz diagnóstico e não substitui avaliação médica. "
    "Em caso de piora, procure uma unidade de saúde."
)
st.info("Em construção: depende do índice vetorial publicado no Hub.")
