"""Dashboard de arboviroses (AGENTS.md, seção 7.1).

Dengue, chikungunya e zika: casos prováveis por semana, no município de residência.
Febre amarela: casos confirmados por período de monitoramento (julho a junho), no
município provável de infecção. As duas visões dividem os filtros de doença e local.
"""

from datetime import date

import streamlit as st

from app import data, graficos, painel
from app.painel import BRASIL, DOENCAS, FEBRE_AMARELA, UFS

GRAFICO = {"width": "stretch", "config": {"displayModeBar": False}}


def _tema() -> str | None:
    """Modo do tema ativo (`light` ou `dark`), para as cores dos gráficos."""
    try:
        return st.context.theme.type
    except AttributeError:
        return None


def _escolher(coluna, rotulo: str, opcoes: list, chave: str):
    """Selectbox que mantém a escolha anterior quando a lista de opções muda, se possível."""
    preferido = st.session_state.get(chave)
    escolha = coluna.selectbox(
        rotulo, opcoes, index=opcoes.index(preferido) if preferido in opcoes else 0
    )
    st.session_state[chave] = escolha
    return escolha


def painel_arboviroses(doenca, uf, municipio, local, nome_local, coluna_ano) -> None:
    """Dengue, chikungunya e zika: série semanal de casos prováveis."""
    info = data.metadados("arboviroses")["doencas"][doenca]
    ultimo_ano = int(info["ultima_semana"].split("-")[0])
    ano = _escolher(
        coluna_ano, "Ano", list(range(ultimo_ano, info["ano_inicial"] - 1, -1)), "ano_preferido"
    )
    municipios, populacao = data.municipios(), data.populacao()
    casos_da_uf = data.casos_uf(doenca, uf) if uf != BRASIL else None
    serie = painel.serie_do_local(data.resumo_arboviroses(), casos_da_uf, doenca, local)
    numeros = painel.indicadores(serie, ano, painel.populacao_do_local(populacao, local))

    st.subheader(f"{DOENCAS[doenca]} · {nome_local} · {ano}")
    tile_casos, tile_incidencia, tile_pico, tile_obitos = st.columns(4)
    variacao = None
    if numeros.variacao is not None:
        variacao = (
            f"{painel.formatar_variacao(numeros.variacao)} em relação a {ano - 1}, "
            f"até a semana {numeros.semana_comparada}"
        )
    tile_casos.metric(
        "Casos prováveis",
        painel.formatar_inteiro(numeros.casos),
        delta=variacao,
        delta_color="inverse",
        help="Notificações não descartadas: confirmadas, em investigação ou inconclusivas. "
        "A comparação com o ano anterior usa só semanas completas.",
    )
    tile_incidencia.metric(
        "Incidência por 100 mil habitantes",
        painel.formatar_decimal(numeros.incidencia) if numeros.incidencia is not None else "—",
        help="Casos prováveis do ano por 100 mil habitantes, com a população estimada de 2022.",
    )
    tile_pico.metric(
        "Semana de pico",
        f"Semana {numeros.semana_pico}" if numeros.semana_pico else "—",
        help=f"Semana com mais casos no ano: {painel.formatar_inteiro(numeros.casos_no_pico)}.",
    )
    tile_obitos.metric(
        "Óbitos pela doença",
        painel.formatar_inteiro(numeros.obitos),
        help="Óbitos pelo agravo entre os casos prováveis. A investigação de um óbito leva "
        "meses, então o ano corrente tende a ficar abaixo do número final.",
    )

    if doenca == "zika" and ano <= 2016:
        st.info(
            "A zika passou a ser de notificação compulsória em 2016. Em 2015 e 2016, os "
            "registros subestimam a epidemia, sobretudo no Nordeste."
        )

    faixa, anos_da_faixa = painel.faixa_historica(serie, ano)
    st.markdown("**Casos por semana**")
    if anos_da_faixa:
        st.caption(
            f"Linha verde: {ano}. Em marrom, a mediana de {anos_da_faixa[0]} a "
            f"{anos_da_faixa[-1]} e a faixa onde ficou metade desses anos em cada semana."
        )
    st.plotly_chart(
        graficos.grafico_semanal(serie, ano, faixa, anos_da_faixa, tema=_tema()), **GRAFICO
    )

    st.markdown("**Casos por ano**")
    st.plotly_chart(
        graficos.grafico_anual(painel.totais_anuais(serie), ano, tema=_tema()), **GRAFICO
    )

    if casos_da_uf is not None and municipio is None:
        st.markdown(f"**Municípios de {UFS[uf][1]} em {ano}**, do maior para o menor coeficiente")
        st.dataframe(
            painel.ranking_municipios(casos_da_uf, ano, municipios, populacao),
            hide_index=True,
            width="stretch",
            column_config={
                "municipio": "Município",
                "casos": st.column_config.NumberColumn("Casos prováveis", format="localized"),
                "incidencia": st.column_config.NumberColumn(
                    "Incidência por 100 mil hab.", format="%.1f"
                ),
                "obitos": st.column_config.NumberColumn("Óbitos", format="localized"),
                "populacao": st.column_config.NumberColumn("População (2022)", format="localized"),
            },
        )

    with st.expander("Ver os dados semanais em tabela"):
        tabela = serie[serie["ano_epi"] == ano].merge(faixa, on="semana_epi", how="left")
        colunas = [
            "semana_epi",
            "inicio_semana",
            "casos_provaveis",
            "obitos",
            "mediana",
            "semana_incompleta",
        ]
        st.dataframe(
            tabela[colunas],
            hide_index=True,
            width="stretch",
            column_config={
                "semana_epi": "Semana",
                "inicio_semana": st.column_config.DateColumn("Início", format="DD/MM/YYYY"),
                "casos_provaveis": "Casos prováveis",
                "obitos": "Óbitos",
                "mediana": st.column_config.NumberColumn(
                    f"Mediana {anos_da_faixa[0]}–{anos_da_faixa[-1]}"
                    if anos_da_faixa
                    else "Mediana",
                    format="%.0f",
                ),
                "semana_incompleta": "Incompleta",
            },
        )

    corte = date.fromisoformat(info["corte"]).strftime("%d/%m/%Y")
    st.caption(
        f"Casos prováveis no município de residência, por semana de início dos sintomas. "
        f"Dados do SINAN publicados pelo DATASUS em {corte}. As últimas semanas antes dessa "
        "data ficam incompletas por causa do atraso de notificação e aparecem sombreadas. "
        "Dados de vigilância estão sujeitos a revisão e não substituem os boletins oficiais "
        "do Ministério da Saúde. População: estimativa do IBGE para 2022."
    )


def painel_febre_amarela(uf, municipio, local, nome_local, coluna_periodo) -> None:
    """Febre amarela: casos confirmados por período de monitoramento e mês."""
    try:
        tabela, info = data.febre_amarela(), data.metadados("febre_amarela")
    except (FileNotFoundError, OSError):
        st.info("Os dados de febre amarela ainda não foram publicados.")
        return
    periodos = painel.periodos_de_monitoramento(info["periodos"][0], info["periodos"][-1])
    periodo = _escolher(coluna_periodo, "Período", periodos[::-1], "periodo_preferido")
    dados = painel.febre_amarela_do_local(tabela, local)
    numeros = painel.indicadores_febre_amarela(dados, periodo)

    st.subheader(f"Febre amarela · {nome_local} · {periodo}")
    tile_casos, tile_obitos, tile_letalidade, tile_municipios = st.columns(4)
    tile_casos.metric(
        "Casos confirmados",
        painel.formatar_inteiro(numeros.casos),
        help="Casos humanos confirmados com início dos sintomas no período de monitoramento, "
        "de julho a junho.",
    )
    tile_obitos.metric("Óbitos", painel.formatar_inteiro(numeros.obitos))
    tile_letalidade.metric(
        "Letalidade",
        f"{painel.formatar_decimal(numeros.letalidade * 100)}%"
        if numeros.letalidade is not None
        else "—",
        help="Óbitos entre os casos confirmados do período.",
    )
    tile_municipios.metric(
        "Municípios com casos",
        painel.formatar_inteiro(numeros.municipios),
        help="Municípios prováveis de infecção com ao menos um caso no período.",
    )

    st.markdown("**Casos por período de monitoramento**")
    st.caption(f"Em verde, {periodo}. Cada período vai de julho a junho do ano seguinte.")
    st.plotly_chart(
        graficos.grafico_periodos(
            painel.febre_amarela_por_periodo(dados, periodos), periodo, tema=_tema()
        ),
        **GRAFICO,
    )

    por_mes = painel.febre_amarela_por_mes(dados, periodo)
    st.markdown(f"**Casos por mês de início dos sintomas, {periodo}**")
    st.plotly_chart(graficos.grafico_meses(por_mes, tema=_tema()), **GRAFICO)

    if municipio is None:
        lista = painel.municipios_febre_amarela(dados, periodo, data.municipios())
        if not lista.empty:
            st.markdown(f"**Municípios prováveis de infecção em {periodo}**")
            st.dataframe(
                lista,
                hide_index=True,
                width="stretch",
                column_config={
                    "municipio": "Município",
                    "uf": "UF",
                    "casos": st.column_config.NumberColumn("Casos", format="localized"),
                    "obitos": st.column_config.NumberColumn("Óbitos", format="localized"),
                },
            )

    with st.expander("Ver os casos por mês em tabela"):
        st.dataframe(
            por_mes[["rotulo", "casos", "obitos"]],
            hide_index=True,
            width="stretch",
            column_config={"rotulo": "Mês", "casos": "Casos", "obitos": "Óbitos"},
        )

    extraido = date.fromisoformat(info["extraido_em"][:10]).strftime("%d/%m/%Y")
    st.caption(
        "Casos humanos confirmados de febre amarela no município provável de infecção, em "
        "geral área de mata, e não no de residência. Fonte: API de Dados Abertos do "
        f"Ministério da Saúde, consultada em {extraido}. Casos sem município ou sem mês "
        "entram nos totais, mas não na tabela de municípios nem no gráfico por mês."
    )


st.title("📊 Arboviroses no Brasil")
st.caption("Dengue, chikungunya, zika e febre amarela nos dados abertos do Ministério da Saúde.")

coluna_doenca, coluna_uf, coluna_municipio, coluna_ano = st.columns([1, 1.3, 1.7, 0.8])
doenca = coluna_doenca.selectbox("Doença", list(DOENCAS), format_func=DOENCAS.get)
uf = coluna_uf.selectbox(
    "Local",
    [BRASIL, *sorted(UFS, key=lambda c: UFS[c][1])],
    format_func=lambda c: "Brasil" if c == BRASIL else UFS[c][1],
)
if uf == BRASIL:
    coluna_municipio.selectbox(
        "Município", ["todos"], format_func=lambda _: "Escolha uma UF primeiro", disabled=True
    )
    municipio, nome_local = None, "Brasil"
else:
    municipios = data.municipios()
    da_uf = municipios[municipios["uf_codigo"] == uf].sort_values("nome")
    nomes = dict(zip(da_uf["codigo_ibge6"], da_uf["nome"], strict=True))
    municipio = coluna_municipio.selectbox(
        "Município",
        [None, *nomes],
        format_func=lambda c: "Todos os municípios da UF" if c is None else nomes[c],
    )
    nome_local = f"{nomes[municipio]} ({UFS[uf][0]})" if municipio else UFS[uf][1]

if doenca == FEBRE_AMARELA:
    painel_febre_amarela(uf, municipio, municipio or uf, nome_local, coluna_ano)
else:
    painel_arboviroses(doenca, uf, municipio, municipio or uf, nome_local, coluna_ano)
