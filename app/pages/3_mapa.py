"""Mapa das UBS mais próximas a partir do CEP (AGENTS.md, seção 7.3)."""

import streamlit as st
from streamlit_folium import st_folium

from app import data, geo


class _SemDetalhe(Exception):
    """Sinaliza falha no CNES sem deixar o st.cache_data guardar o resultado vazio."""


@st.cache_data(ttl=86400, show_spinner=False)
def _localizar(cep: str) -> geo.Localizacao:
    # Exceções não entram no cache: um CEP que falhou por instabilidade é tentado de novo.
    return geo.geocodificar_cep(cep)


@st.cache_data(ttl=86400, show_spinner=False)
def _detalhe_cacheado(cnes: str) -> dict:
    detalhe = geo.detalhar_estabelecimento(cnes)
    if detalhe is None:
        raise _SemDetalhe(cnes)
    return detalhe


def _detalhar(cnes: str) -> dict | None:
    try:
        return _detalhe_cacheado(cnes)
    except _SemDetalhe:
        return None


st.title("Unidades básicas de saúde mais próximas")
st.caption(
    "Informe o CEP para ver no mapa as UBS mais próximas. A distância é em linha reta; "
    "o trajeto real pode ser maior."
)

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
    st.caption(f"Coordenadas do CEP obtidas via {origem.fonte}.")
    st_folium(
        geo.mapa_ubs(origem, proximas, detalhes),
        height=480,
        use_container_width=True,
        returned_objects=[],
    )

    for posicao, ubs in enumerate(proximas.to_dict("records"), start=1):
        detalhe = detalhes.get(ubs["cnes"]) or {}
        with st.container(border=True):
            st.markdown(
                f"**{posicao}. {detalhe.get('nome') or ubs['nome']}** · "
                f"{ubs['distancia_km']:.1f} km"
            )
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
            st.link_button(
                "Ver rota a pé", geo.link_rota(origem, ubs["latitude"], ubs["longitude"])
            )

    st.caption(
        "Fontes: cadastro de UBS e CNES (Ministério da Saúde, dados abertos); "
        "mapa © colaboradores do OpenStreetMap."
    )
