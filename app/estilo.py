"""Tamanho de fonte das páginas, maior que o padrão do Streamlit para leitura confortável.

O mapa estabeleceu o padrão; o dashboard o segue e acrescenta as regras dos elementos
que só ele usa. Uma tag <style> sozinha no `st.html` não ocupa espaço na página.
"""

import streamlit as st

# Texto corrido, rótulos de campos, legendas e caixas de aviso (o título fica como está).
BASE = """
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { font-size: 1.15rem !important; }
[data-testid="stCaptionContainer"] p { font-size: 1.05rem !important; }
[data-testid="stTextInputRootElement"] input { font-size: 1.15rem !important; }
/* Valor e variação dos indicadores herdam o tamanho do próprio bloco; sem isto, a regra
   acima os deixaria do tamanho do texto corrido. */
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"] p {
    font-size: inherit !important;
}
"""

# Seletores, a lista de opções aberta e a variação dos indicadores (nomes do Streamlit
# 1.66). A lista aberta é virtualizada, com altura de linha fixa: fonte um pouco menor
# que a dos seletores para não cortar as opções.
DASHBOARD = """
[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] input { font-size: 1.15rem !important; }
[data-testid="stSelectboxVirtualDropdown"] div { font-size: 1.05rem !important; }
[data-testid="stMetricDelta"] { font-size: 1.05rem !important; }
/* A variação quebra linha em colunas estreitas, em vez de terminar em reticências. */
[data-testid="stMetricDelta"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
"""


def aplicar(*regras: str) -> None:
    """Injeta na página as regras de fonte pedidas (a base, se nenhuma for passada)."""
    st.html("<style>" + "".join(regras or (BASE,)) + "</style>")
