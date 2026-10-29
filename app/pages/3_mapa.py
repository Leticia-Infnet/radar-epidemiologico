"""Mapa das UBS mais próximas a partir do CEP, com rota a pé (AGENTS.md, seção 7.3)."""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app import data, geo


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
def _rota_cacheada(origem: geo.Localizacao, lat: float, lon: float) -> geo.Rota:
    rota = geo.rota_a_pe(origem, lat, lon)
    if rota is None:
        raise _SemResultado((lat, lon))
    return rota


def _rota(origem: geo.Localizacao, lat: float, lon: float) -> geo.Rota | None:
    try:
        return _rota_cacheada(origem, lat, lon)
    except _SemResultado:
        return None


def _ordenar_a_pe(origem: geo.Localizacao, proximas: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta distância e tempo a pé e ordena por eles; sem rota, vale a linha reta."""
    destinos = list(proximas[["latitude", "longitude"]].itertuples(index=False, name=None))
    a_pe = geo.distancias_a_pe(origem, destinos)
    proximas = proximas.assign(
        distancia_pe_m=[d[0] if d else float("nan") for d in a_pe],
        duracao_pe_s=[d[1] if d else float("nan") for d in a_pe],
    )
    return proximas.sort_values(["distancia_pe_m", "distancia_km"], na_position="last").reset_index(
        drop=True
    )


st.title("Unidades básicas de saúde mais próximas")
st.caption("Informe o CEP para ver no mapa as UBS mais próximas e o caminho a pé até elas.")

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
        with st.spinner("Calculando distâncias a pé e consultando o CNES..."):
            proximas = _ordenar_a_pe(origem, proximas)
            detalhes = {cnes: _detalhar(cnes) for cnes in proximas["cnes"]}
        st.session_state["mapa_ubs"] = (origem, proximas, detalhes)

if resultado := st.session_state.get("mapa_ubs"):
    origem, proximas, detalhes = resultado
    registros = proximas.to_dict("records")
    nomes = {
        u["cnes"]: geo.texto((detalhes.get(u["cnes"]) or {}).get("nome")) or u["nome"]
        for u in registros
    }
    rotulos = {
        u["cnes"]: f"{i}. {nomes[u['cnes']]} · {geo.distancia_resumida(u)}"
        for i, u in enumerate(registros, start=1)
    }

    st.markdown(f"**Ponto de partida:** {origem.endereco or 'CEP informado'}")
    destino = st.radio(
        "Mostrar o caminho a pé até:",
        options=list(rotulos),
        format_func=rotulos.get,
    )
    alvo = next(u for u in registros if u["cnes"] == destino)
    with st.spinner("Traçando a rota..."):
        rota = _rota(origem, alvo["latitude"], alvo["longitude"])

    st_folium(
        geo.mapa_ubs(origem, proximas, detalhes, rota=rota, destino=destino),
        height=480,
        use_container_width=True,
        returned_objects=[],
    )
    if rota:
        st.caption(
            f"Caminho a pé: {geo.formatar_distancia(rota.distancia_m)}, cerca de "
            f"{geo.formatar_duracao(rota.duracao_s)}. Trajeto estimado com dados do "
            "OpenStreetMap; confira calçadas, travessias e segurança no local."
        )
    else:
        st.info("Não foi possível calcular o caminho a pé agora.")
        st.link_button(
            "Abrir a rota no OpenStreetMap",
            geo.link_rota(origem, alvo["latitude"], alvo["longitude"]),
        )
    if alvo["coordenada_aproximada"]:
        st.caption("A posição desta unidade no cadastro é aproximada; a rota também é.")

    for posicao, ubs in enumerate(registros, start=1):
        detalhe = detalhes.get(ubs["cnes"]) or {}
        with st.container(border=True):
            st.markdown(f"**{posicao}. {nomes[ubs['cnes']]}**")
            st.caption(geo.descrever_distancia(ubs))
            endereco = geo.texto(detalhe.get("endereco")) or geo.texto(ubs["logradouro"])
            bairro = geo.texto(detalhe.get("bairro")) or geo.texto(ubs["bairro"])
            if local := ", ".join(p for p in (endereco, bairro) if p):
                st.write(local)
            if detalhe.get("telefone"):
                st.write(f"Telefone: {detalhe['telefone']}")
            if detalhe.get("turno"):
                st.write(detalhe["turno"].capitalize())
            if ubs["coordenada_aproximada"]:
                st.caption("A posição desta unidade no mapa é aproximada.")
            if not detalhe:
                st.caption("Detalhes do CNES indisponíveis no momento.")

    st.caption(
        "Fontes: cadastro de UBS e CNES (Ministério da Saúde, dados abertos); mapa © "
        "colaboradores do OpenStreetMap e CARTO; rotas a pé calculadas pelo OSRM da FOSSGIS."
    )
