"""Estimativa de tendência de casos prováveis (AGENTS.md, seção 7.2).

O modelo é treinado no CI (`ml/train.py`) e publica as estimativas prontas no Hub: esta
página só as lê e compara com a série observada. Nada é treinado nem calculado em request.
"""

from datetime import date

import streamlit as st

from app import data, estilo, filtros, graficos, painel, tendencia
from app.painel import UFS

GRAFICO = {"width": "stretch", "config": {"displayModeBar": False}}


def _data(iso: str) -> str:
    return date.fromisoformat(iso[:10]).strftime("%d/%m/%Y")


def _como_e_feita(info: dict, info_doenca: dict, modelo: str, desempenho) -> None:
    """Explicação do método e erro do modelo no backtest do local."""
    incompletas = info_doenca["semanas_incompletas"]
    st.markdown(
        f"""
- **Modelo**: {tendencia.MODELOS[modelo].lower()}, treinado em {_data(info["treinado_em"])}
  com os dados publicados até {_data(info_doenca["corte"])} e refeito a cada atualização
  semanal. Há um modelo para o Brasil e as UFs e outro para os municípios.
- **O que ele olha**: as últimas semanas do próprio local, o que aconteceu ali na mesma
  época nos anos anteriores e a época do ano. Não usa clima, sorotipos em circulação nem
  dados de outros locais.
- **Por que começa antes do fim da curva**: as {incompletas} últimas semanas chegam
  incompletas pelo atraso de notificação. A estimativa parte da última semana completa,
  cobre essas {incompletas} semanas e avança mais {info["semanas_a_frente"]}.
- **De onde vêm os intervalos**: dos erros do modelo em semanas passadas que ele estimou
  sem ver o resultado, de {info_doenca["primeiro_ano_teste"]} em diante. Em cada ano de
  teste o modelo foi treinado só com o que já se conhecia.
- **O dado recente ainda muda, nos dois sentidos**: sobe com as notificações que chegam
  atrasadas e desce quando casos suspeitos são descartados. Por isso o dado parcial de
  uma semana incompleta pode ficar acima ou abaixo da estimativa.
- **Limites**: o teste usa os dados como estão hoje, já revisados; na prática o modelo
  parte de semanas que ainda serão revisadas, e o erro real tende a ser um pouco maior.
  Viradas bruscas, como o início de uma epidemia, são o que o modelo mais erra.
"""
    )
    if desempenho is None:
        return
    st.markdown("**Erro no teste com semanas passadas, neste local**")
    st.dataframe(
        desempenho,
        hide_index=True,
        width="stretch",
        column_config={
            "passo": st.column_config.NumberColumn(
                "Semanas desde a última completa",
                help=f"As {incompletas} primeiras são as semanas incompletas.",
            ),
            "mae": st.column_config.NumberColumn(
                "Erro médio (casos)", format="%.0f", help="Erro absoluto médio por semana."
            ),
            "mase": st.column_config.NumberColumn(
                "MASE do modelo",
                format="%.2f",
                help="Erro do modelo dividido pelo erro típico de repetir a mesma semana do "
                "ano anterior. Abaixo de 1, o modelo erra menos.",
            ),
            "mase_persistencia": st.column_config.NumberColumn(
                "MASE de repetir a última semana", format="%.2f"
            ),
            "mase_sazonal": st.column_config.NumberColumn(
                "MASE de repetir o ano anterior", format="%.2f"
            ),
            "cobertura_90": st.column_config.NumberColumn(
                "Acerto do intervalo de 90%",
                format="percent",
                help="Fração das semanas de teste em que o valor real caiu dentro do "
                "intervalo de 90%. O esperado é perto de 90%.",
            ),
        },
    )


estilo.aplicar(estilo.BASE, estilo.DASHBOARD)
st.title("📈 Estimativa de tendência")
st.caption(
    "Estimativa estatística da tendência dos casos prováveis nas próximas semanas, com "
    "intervalo. **Não é uma previsão oficial** do Ministério da Saúde."
)

coluna_doenca, coluna_uf, coluna_municipio = st.columns([1, 1.3, 1.7])
doenca = coluna_doenca.selectbox(
    "Doença",
    list(tendencia.DOENCAS),
    format_func=tendencia.DOENCAS.get,
    help="Zika e febre amarela ficam de fora: têm poucos casos por semana para o modelo.",
)
uf, municipio, nome_local = filtros.escolher_local(coluna_uf, coluna_municipio)

estimativas = data.estimativas()
if estimativas is None:
    st.info("As estimativas de tendência ainda não foram publicadas. Volte em breve.")
    st.stop()
info = data.metadados("previsao")
info_doenca = info["doencas"][doenca]

local = municipio or uf
estimativa = tendencia.estimativa_do_local(estimativas, doenca, local)
if estimativa.empty and municipio:
    criterio = info["criterio_de_municipios"]
    st.info(
        f"{nome_local} tem poucos casos de {tendencia.DOENCAS[doenca].lower()} para uma "
        "estimativa própria: só entram municípios com média de pelo menos "
        f"{painel.formatar_inteiro(criterio['media_minima_semanal'])} casos por semana nos "
        f"últimos {criterio['semanas'] // 52} anos. Abaixo, a estimativa de {UFS[uf][1]}."
    )
    local, nome_local = uf, UFS[uf][1]
    estimativa = tendencia.estimativa_do_local(estimativas, doenca, local)
if estimativa.empty:
    st.info(f"Ainda não há estimativa de {tendencia.DOENCAS[doenca].lower()} para {nome_local}.")
    st.stop()

do_municipio = len(local) == 6
casos_da_uf = data.casos_uf(doenca, uf) if do_municipio else None
serie = painel.serie_do_local(data.resumo_arboviroses(), casos_da_uf, doenca, local)
recente = tendencia.serie_recente(serie, estimativa)
numeros = tendencia.tendencia(recente, estimativa)
semana_final = numeros.inicio_semana.strftime("%d/%m")

st.subheader(f"{tendencia.DOENCAS[doenca]} · {nome_local}")
tile_atual, tile_estimativa, tile_intervalo, tile_tendencia = st.columns(4)
tile_atual.metric(
    "Nível atual (casos por semana)",
    painel.formatar_inteiro(numeros.nivel_atual),
    help=f"Média das {tendencia.SEMANAS_DE_REFERENCIA} últimas semanas completas.",
)
tile_estimativa.metric(
    f"Estimativa para a semana de {semana_final}",
    painel.formatar_inteiro(numeros.estimativa),
    delta=f"{painel.formatar_variacao(numeros.variacao)} sobre o nível atual"
    if numeros.variacao is not None
    else None,
    delta_color="off" if numeros.direcao == tendencia.ESTAVEL else "inverse",
    help="Casos prováveis estimados para a última semana do horizonte. É o valor central "
    "do modelo, não uma certeza: veja o intervalo ao lado.",
)
tile_intervalo.metric(
    "Intervalo de 90%",
    f"{painel.formatar_inteiro(numeros.inferior)} a {painel.formatar_inteiro(numeros.superior)}",
    help=f"Faixa em que o número de casos da semana de {semana_final} deve cair em 9 de cada "
    "10 vezes, pelo erro do modelo em semanas passadas.",
)
tile_tendencia.metric(
    "Tendência",
    numeros.direcao,
    help="Alta ou queda só quando a faixa de 50% da estimativa fica inteira acima ou abaixo "
    "do nível atual. Se a faixa inclui o nível atual, a tendência é tratada como estável.",
)

st.markdown("**Casos por semana: observado e estimado**")
st.caption(
    "Linha cheia: semanas completas. Linha tracejada: estimativa, com as faixas de 50% "
    "(mais escura) e 90%. Na área sombreada, o dado ainda está incompleto; os pontos em "
    "marrom mostram o dado parcial já publicado, que ainda será revisado."
)
st.plotly_chart(graficos.grafico_tendencia(recente, estimativa, tema=filtros.tema()), **GRAFICO)

with st.expander("Ver as estimativas em tabela"):
    st.dataframe(
        tendencia.tabela_semanal(recente, estimativa),
        hide_index=True,
        width="stretch",
        column_config={
            "semana_epi": "Semana",
            "inicio_semana": st.column_config.DateColumn("Início", format="DD/MM/YYYY"),
            "estimativa": st.column_config.NumberColumn("Estimativa", format="%.0f"),
            "p05": st.column_config.NumberColumn("90%: de", format="%.0f"),
            "p95": st.column_config.NumberColumn("90%: até", format="%.0f"),
            "p25": st.column_config.NumberColumn("50%: de", format="%.0f"),
            "p75": st.column_config.NumberColumn("50%: até", format="%.0f"),
            "parcial": st.column_config.NumberColumn(
                "Dado parcial",
                help="O que já foi publicado das semanas incompletas. Ainda muda: sobe com "
                "notificações atrasadas e desce com os descartes.",
            ),
        },
    )

with st.expander("Como a estimativa é feita"):
    modelo = info_doenca["grupos"]["municipio" if do_municipio else "agregado"]["modelo"]
    metricas = data.metricas_previsao()
    desempenho = (
        tendencia.desempenho(metricas, doenca, local, modelo) if metricas is not None else None
    )
    _como_e_feita(info, info_doenca, modelo, desempenho)

st.caption(
    "Casos prováveis no município de residência, por semana de início dos sintomas (SINAN, "
    f"DATASUS). Estimativas geradas em {_data(info['treinado_em'])} a partir dos dados "
    f"publicados até {_data(info_doenca['corte'])}. São um apoio para acompanhar a "
    "tendência e não substituem os boletins nem as orientações do Ministério da Saúde e "
    "das secretarias de saúde."
)
