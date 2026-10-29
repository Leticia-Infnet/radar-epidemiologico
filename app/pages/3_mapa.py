"""Mapa das UBS mais próximas a partir do CEP, com rota a pé ou de carro (AGENTS.md, 7.3)."""

import re

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app import data, geo

AVISOS = {
    "a_pe": "Trajeto estimado com dados do OpenStreetMap; confira calçadas, travessias e "
    "segurança no local.",
    "carro": "Tempo estimado sem considerar o trânsito, com dados do OpenStreetMap.",
}


# Fonte maior que o padrão do Streamlit em toda a página, exceto o título. Uma tag
# <style> sozinha no st.html não ocupa espaço na página.
ESTILO_PAGINA = """
<style>
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { font-size: 1.15rem !important; }
[data-testid="stCaptionContainer"] p { font-size: 1.05rem !important; }
[data-testid="stTextInputRootElement"] input { font-size: 1.15rem !important; }
</style>
"""


def _sem_markdown(texto: str) -> str:
    """Escapa caracteres que o Markdown interpretaria em textos vindos do cadastro."""
    return re.sub(r"([\\`*_{}\[\]<>#|~])", r"\\\1", texto)


class _SemResultado(Exception):
    """Sinaliza falha sem deixar o st.cache_data guardar o resultado vazio."""


@st.cache_data(ttl=86400, show_spinner=False)
def _localizar(cep: str) -> geo.Localizacao:
    # Exceções não entram no cache: um CEP que falhou por instabilidade é tentado de novo.
    return geo.geocodificar_cep(cep)


@st.cache_data(ttl=86400, show_spinner=False)
def _detalhe_cacheado(cnes: str) -> dict:
    detalhe = geo.detalhar_estabelecimento(cnes)
    if detalhe is None:
        raise _SemResultado(cnes)
    return detalhe


def _detalhar(cnes: str) -> dict | None:
    try:
        return _detalhe_cacheado(cnes)
    except _SemResultado:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _distancias_cacheadas(
    origem: geo.Localizacao, destinos: tuple[tuple[float, float], ...], modo: str
) -> list:
    resultado = geo.distancias(origem, list(destinos), modo)
    if not any(resultado):
        raise _SemResultado(modo)
    return resultado


@st.cache_data(ttl=86400, show_spinner=False)
def _rota_cacheada(origem: geo.Localizacao, lat: float, lon: float, modo: str) -> geo.Rota:
    achada = geo.rota(origem, lat, lon, modo)
    if achada is None:
        raise _SemResultado((lat, lon, modo))
    return achada


def _rota(origem: geo.Localizacao, lat: float, lon: float, modo: str) -> geo.Rota | None:
    try:
        return _rota_cacheada(origem, lat, lon, modo)
    except _SemResultado:
        return None


def _ordenar(origem: geo.Localizacao, proximas: pd.DataFrame, modo: str) -> pd.DataFrame:
    """Acrescenta distância e tempo pelo modo escolhido e ordena por eles.

    Sem resposta do serviço de rotas, mantém a ordem da linha reta.
    """
    destinos = tuple(proximas[["latitude", "longitude"]].itertuples(index=False, name=None))
    try:
        medidas = _distancias_cacheadas(origem, destinos, modo)
    except _SemResultado:
        medidas = [None] * len(destinos)
    proximas = proximas.assign(
        distancia_rota_m=[m[0] if m else float("nan") for m in medidas],
        duracao_rota_s=[m[1] if m else float("nan") for m in medidas],
    )
    return proximas.sort_values(
        ["distancia_rota_m", "distancia_km"], na_position="last"
    ).reset_index(drop=True)


st.html(ESTILO_PAGINA)
st.title("🗺️ Unidades básicas de saúde mais próximas")
st.caption("Informe o CEP para ver no mapa as UBS mais próximas e o caminho até elas.")

with st.form("busca_ubs"):
    coluna_cep, coluna_k = st.columns([2, 1])
    cep = coluna_cep.text_input("CEP", placeholder="00000-000", max_chars=9)
    quantidade = coluna_k.slider("Quantidade de unidades", min_value=1, max_value=5, value=3)
    buscar = st.form_submit_button("Buscar", type="primary")

if buscar:
    st.session_state.pop("mapa_ubs", None)
    try:
        with st.spinner("Localizando o CEP..."):
            origem = _localizar(geo.normalizar_cep(cep))
    except geo.CepInvalido:
        st.error("Digite um CEP com 8 dígitos, por exemplo 53020-140.")
    except geo.CepNaoEncontrado:
        st.error(
            "Não encontramos a localização deste CEP. CEPs gerais de cidade (terminados em "
            "000) costumam não ter coordenadas; tente o CEP da sua rua."
        )
    else:
        indice, ubs = data.indice_ubs()
        proximas = geo.ubs_mais_proximas(
            indice, ubs, origem.latitude, origem.longitude, k=quantidade
        )
        with st.spinner("Consultando o cadastro das unidades (CNES)..."):
            detalhes = {cnes: _detalhar(cnes) for cnes in proximas["cnes"]}
        st.session_state["mapa_ubs"] = (origem, proximas, detalhes)

if resultado := st.session_state.get("mapa_ubs"):
    origem, proximas, detalhes = resultado
    st.markdown(f"**Ponto de partida:** {origem.endereco or 'CEP informado'}")

    modo = st.radio(
        "Como você vai até a unidade?",
        options=list(geo.MODOS),
        format_func=lambda m: geo.MODOS[m].rotulo.capitalize(),
        horizontal=True,
    )
    rotulo = geo.MODOS[modo].rotulo
    with st.spinner(f"Calculando as distâncias {rotulo}..."):
        proximas = _ordenar(origem, proximas, modo)
    registros = proximas.to_dict("records")
    nomes = {
        u["cnes"]: geo.texto((detalhes.get(u["cnes"]) or {}).get("nome")) or u["nome"]
        for u in registros
    }
    rotulos = {
        u["cnes"]: f"{i}. {nomes[u['cnes']]} · {geo.distancia_resumida(u, modo)}"
        for i, u in enumerate(registros, start=1)
    }
    destino = st.radio("Mostrar o caminho até:", options=list(rotulos), format_func=rotulos.get)
    alvo = next(u for u in registros if u["cnes"] == destino)
    with st.spinner("Traçando a rota..."):
        rota = _rota(origem, alvo["latitude"], alvo["longitude"], modo)

    st_folium(
        geo.mapa_ubs(origem, proximas, detalhes, rota=rota, destino=destino, modo=modo),
        height=480,
        use_container_width=True,
        returned_objects=[],
    )
    if rota:
        st.caption(
            f"Caminho {rotulo}: {geo.formatar_distancia(rota.distancia_m)}, cerca de "
            f"{geo.formatar_duracao(rota.duracao_s)}. {AVISOS[modo]}"
        )
    else:
        st.info(f"Não foi possível calcular o caminho {rotulo} agora.")
        st.link_button(
            "Abrir a rota no OpenStreetMap",
            geo.link_rota(origem, alvo["latitude"], alvo["longitude"], modo),
        )
    if alvo["coordenada_aproximada"]:
        st.caption("A posição desta unidade no cadastro é aproximada; a rota também é.")

    for posicao, ubs in enumerate(registros, start=1):
        detalhe = detalhes.get(ubs["cnes"]) or {}
        with st.container(border=True):
            st.markdown(f"**{posicao}. {_sem_markdown(nomes[ubs['cnes']])}**")
            # Dois espaços antes da quebra ("  \n") fazem o Markdown mudar de linha.
            st.markdown(
                "  \n".join(
                    f"**{rotulo}:** {_sem_markdown(valor)}"
                    for rotulo, valor in geo.campos_unidade(ubs, detalhe, modo)
                )
            )
            if ubs["coordenada_aproximada"]:
                st.caption("A posição desta unidade no mapa é aproximada.")
            if not detalhe:
                st.caption("Detalhes do CNES indisponíveis no momento.")

    st.caption(
        "Fontes: cadastro de UBS e CNES (Ministério da Saúde, dados abertos); mapa © "
        "colaboradores do OpenStreetMap; rotas calculadas pelo OSRM da FOSSGIS."
    )
