"""Filtros e utilidades que o dashboard e a página de tendência dividem."""

import streamlit as st

from app import data
from app.painel import BRASIL, UFS


def tema() -> str | None:
    """Modo do tema ativo (`light` ou `dark`), para as cores dos gráficos."""
    try:
        return st.context.theme.type
    except AttributeError:
        return None


def escolher(coluna, rotulo: str, opcoes: list, chave: str):
    """Selectbox que mantém a escolha anterior quando a lista de opções muda, se possível."""
    preferido = st.session_state.get(chave)
    escolha = coluna.selectbox(
        rotulo, opcoes, index=opcoes.index(preferido) if preferido in opcoes else 0
    )
    st.session_state[chave] = escolha
    return escolha


def escolher_local(coluna_uf, coluna_municipio) -> tuple[str, str | None, str]:
    """Seletores de local (Brasil ou UF) e de município da UF.

    Devolve o código do local (`BR` ou a UF), o código do município (None para a UF
    inteira ou o Brasil) e o nome a exibir.
    """
    uf = coluna_uf.selectbox(
        "Local",
        [BRASIL, *sorted(UFS, key=lambda c: UFS[c][1])],
        format_func=lambda c: "Brasil" if c == BRASIL else UFS[c][1],
    )
    if uf == BRASIL:
        coluna_municipio.selectbox(
            "Município", ["todos"], format_func=lambda _: "Escolha uma UF primeiro", disabled=True
        )
        return uf, None, "Brasil"
    municipios = data.municipios()
    da_uf = municipios[municipios["uf_codigo"] == uf].sort_values("nome")
    nomes = dict(zip(da_uf["codigo_ibge6"], da_uf["nome"], strict=True))
    municipio = coluna_municipio.selectbox(
        "Município",
        [None, *nomes],
        format_func=lambda c: "Todos os municípios da UF" if c is None else nomes[c],
    )
    return uf, municipio, f"{nomes[municipio]} ({UFS[uf][0]})" if municipio else UFS[uf][1]
