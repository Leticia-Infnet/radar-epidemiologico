import pandas as pd
import pytest

from app import graficos, painel
from etl.transform.calendario import calendario


def serie_sintetica(casos_por_ano: dict[int, int], incompletas: int = 0) -> pd.DataFrame:
    """Série semanal com `casos` por semana em cada ano; as últimas semanas incompletas."""
    tabela = calendario(min(casos_por_ano), max(casos_por_ano))
    tabela["casos_provaveis"] = tabela["ano_epi"].map(casos_por_ano)
    tabela["obitos"] = 0
    tabela["semana_incompleta"] = False
    if incompletas:
        tabela.loc[tabela.index[-incompletas:], "semana_incompleta"] = True
    return tabela.rename(columns={"inicio": "inicio_semana"})


RESUMO = pd.concat(
    [
        serie_sintetica({2024: 10, 2025: 5}).assign(doenca="dengue", local="BR"),
        serie_sintetica({2024: 3, 2025: 1}).assign(doenca="dengue", local="26"),
        serie_sintetica({2024: 7, 2025: 7}).assign(doenca="zika", local="BR"),
    ]
)
CASOS_PE = pd.concat(
    [
        serie_sintetica({2024: 2, 2025: 1}).assign(codigo_ibge6="260960"),
        serie_sintetica({2024: 1, 2025: 0}).assign(codigo_ibge6="261160"),
    ]
)
CASOS_PE["codigo_ibge6"] = CASOS_PE["codigo_ibge6"].astype("category")
MUNICIPIOS = pd.DataFrame(
    {
        "codigo_ibge6": ["260960", "261160"],
        "nome": ["Olinda", "Recife"],
        "uf_codigo": ["26", "26"],
        "uf_sigla": ["PE", "PE"],
    }
)
POPULACAO = pd.DataFrame(
    {"codigo_ibge6": ["260960", "261160", "120001"], "populacao": [350_000, 1_500_000, 14_000]}
)


def test_serie_do_local():
    brasil = painel.serie_do_local(RESUMO, None, "dengue", "BR")
    assert brasil["casos_provaveis"].sum() == 52 * 10 + 53 * 5
    assert list(brasil.columns) == painel.COLUNAS_SERIE
    olinda = painel.serie_do_local(RESUMO, CASOS_PE, "dengue", "260960")
    assert olinda["casos_provaveis"].sum() == 52 * 2 + 53
    with pytest.raises(ValueError):
        painel.serie_do_local(RESUMO, None, "dengue", "260960")


@pytest.mark.parametrize(
    ("local", "esperado"), [("BR", 1_864_000), ("26", 1_850_000), ("260960", 350_000), ("35", None)]
)
def test_populacao_do_local(local, esperado):
    assert painel.populacao_do_local(POPULACAO, local) == esperado


def test_indicadores_comparam_so_semanas_completas():
    serie = serie_sintetica({2024: 10, 2025: 20}, incompletas=3)
    serie.loc[(serie["ano_epi"] == 2025) & (serie["semana_epi"] == 30), "casos_provaveis"] = 90
    serie.loc[serie["ano_epi"] == 2025, "obitos"] = 1
    numeros = painel.indicadores(serie, 2025, populacao=1_000_000)
    assert numeros.casos == 53 * 20 + 70  # 2025 tem 53 semanas
    assert numeros.obitos == 53
    assert numeros.incidencia == pytest.approx(numeros.casos / 10)
    # Semanas 51 a 53 de 2025 incompletas: compara 1 a 50 com 1 a 50 de 2024.
    assert numeros.semana_comparada == 50
    assert numeros.variacao == pytest.approx((49 * 20 + 90) / (50 * 10) - 1)
    assert (numeros.semana_pico, numeros.casos_no_pico) == (30, 90)
    assert numeros.tem_semanas_incompletas


def test_indicadores_sem_base_suficiente_nem_populacao():
    numeros = painel.indicadores(serie_sintetica({2024: 0, 2025: 3}), 2025, populacao=None)
    assert numeros.variacao is None  # ano anterior sem casos
    assert numeros.incidencia is None
    vazio = painel.indicadores(serie_sintetica({2024: 0, 2025: 0}), 2025, populacao=10)
    assert (vazio.casos, vazio.semana_pico, vazio.incidencia) == (0, None, 0.0)
    primeiro = painel.indicadores(serie_sintetica({2024: 5}), 2024, populacao=10)
    assert primeiro.variacao is None  # não há ano anterior


def test_faixa_historica_usa_os_anos_anteriores():
    serie = serie_sintetica({2020: 1, 2021: 2, 2022: 3, 2023: 4, 2024: 100})
    faixa, anos = painel.faixa_historica(serie, 2024, anos=3)
    assert anos == [2021, 2022, 2023]
    semana_1 = faixa[faixa["semana_epi"] == 1].iloc[0]
    assert (semana_1["mediana"], semana_1["p25"], semana_1["p75"]) == (3, 2.5, 3.5)
    assert painel.faixa_historica(serie, 2020)[1] == []


def test_totais_anuais():
    totais = painel.totais_anuais(serie_sintetica({2024: 1, 2025: 2}, incompletas=2))
    assert totais.to_dict("list") == {
        "ano_epi": [2024, 2025],
        "casos_provaveis": [52, 106],
        "obitos": [0, 0],
        "incompleto": [False, True],
    }


def test_ranking_por_incidencia():
    ranking = painel.ranking_municipios(CASOS_PE, 2024, MUNICIPIOS, POPULACAO)
    assert ranking["municipio"].tolist() == ["Olinda", "Recife"]
    assert ranking.loc[0, "casos"] == 104
    assert ranking.loc[0, "incidencia"] == pytest.approx(104 / 3.5)


@pytest.mark.parametrize(
    ("funcao", "valor", "esperado"),
    [
        (painel.formatar_inteiro, 6564924, "6.564.924"),
        (painel.formatar_decimal, 1234.56, "1.234,6"),
        (painel.formatar_variacao, -0.7064, "-70,6%"),
        (painel.formatar_variacao, 0.25, "+25,0%"),
    ],
)
def test_formatos_brasileiros(funcao, valor, esperado):
    assert funcao(valor) == esperado


def test_grafico_semanal_destaca_o_ano_e_as_semanas_incompletas():
    serie = serie_sintetica({2024: 10, 2025: 20}, incompletas=4)
    faixa, anos = painel.faixa_historica(serie, 2025)
    figura = graficos.grafico_semanal(serie, 2025, faixa, anos)
    nomes = [t.name for t in figura.data]
    assert nomes == [None, "Faixa típica 2024", "Mediana 2024", "2025"]
    assert figura.data[-1].line.color == graficos.PALETAS["light"].foco
    escuro = graficos.grafico_semanal(serie, 2025, faixa, anos, tema="dark")
    assert escuro.data[-1].line.color == graficos.PALETAS["dark"].foco
    assert figura.layout.separators == ",."
    (sombra,) = figura.layout.shapes
    assert (sombra.x0, sombra.x1) == (49.5, 53.5)


def test_grafico_semanal_sem_historico():
    serie = serie_sintetica({2024: 10})
    figura = graficos.grafico_semanal(serie, 2024, *painel.faixa_historica(serie, 2024))
    assert [t.name for t in figura.data] == ["2024"]
    assert not figura.layout.shapes


def test_grafico_anual_destaca_o_ano_escolhido():
    totais = painel.totais_anuais(serie_sintetica({2023: 1, 2024: 2, 2025: 3}, incompletas=1))
    (barras,) = graficos.grafico_anual(totais, 2024).data
    cores = graficos.PALETAS["light"]
    assert list(barras.marker.color) == [cores.referencia, cores.foco, cores.referencia]
    assert list(barras.customdata) == ["", "", " (ano em andamento)"]


FEBRE_AMARELA = pd.DataFrame(
    {
        "periodo": ["2016/2017", "2016/2017", "2016/2017", "2018/2019", "2018/2019"],
        "ano_inicio_periodo": [2016, 2016, 2016, 2018, 2018],
        "ano": [2016, 2017, 2017, 2019, None],
        "mes": [12, 1, 1, 2, None],
        "codigo_ibge6": ["310620", "310620", "320130", None, "350010"],
        "uf_codigo": ["31", "31", "32", "35", "35"],
        "casos": [3, 5, 2, 1, 4],
        "obitos": [1, 2, 0, 1, 2],
    }
)
MUNICIPIOS_FA = pd.DataFrame(
    {
        "codigo_ibge6": ["310620", "320130", "350010"],
        "nome": ["Belo Horizonte", "Colatina", "Adamantina"],
        "uf_sigla": ["MG", "ES", "SP"],
    }
)


def test_febre_amarela_do_local():
    assert painel.febre_amarela_do_local(FEBRE_AMARELA, "BR")["casos"].sum() == 15
    assert painel.febre_amarela_do_local(FEBRE_AMARELA, "35")["casos"].sum() == 5
    assert painel.febre_amarela_do_local(FEBRE_AMARELA, "310620")["casos"].sum() == 8


def test_periodos_incluem_os_sem_casos():
    assert painel.periodos_de_monitoramento("2016/2017", "2018/2019") == [
        "2016/2017",
        "2017/2018",
        "2018/2019",
    ]


def test_indicadores_febre_amarela():
    numeros = painel.indicadores_febre_amarela(FEBRE_AMARELA, "2016/2017")
    assert (numeros.casos, numeros.obitos, numeros.municipios) == (10, 3, 2)
    assert numeros.letalidade == pytest.approx(0.3)
    vazio = painel.indicadores_febre_amarela(FEBRE_AMARELA, "2017/2018")
    assert (vazio.casos, vazio.letalidade, vazio.municipios) == (0, None, 0)


def test_febre_amarela_por_periodo_e_por_mes():
    periodos = painel.periodos_de_monitoramento("2016/2017", "2018/2019")
    por_periodo = painel.febre_amarela_por_periodo(FEBRE_AMARELA, periodos)
    assert por_periodo["casos"].tolist() == [10, 0, 5]
    por_mes = painel.febre_amarela_por_mes(FEBRE_AMARELA, "2016/2017")
    assert por_mes["rotulo"].tolist()[:7] == [
        "jul/16", "ago/16", "set/16", "out/16", "nov/16", "dez/16", "jan/17"
    ]  # fmt: skip
    assert por_mes.set_index("mes").loc[[12, 1], "casos"].tolist() == [3, 7]
    # O caso sem mês de 2018/2019 fica de fora do gráfico por mês.
    assert painel.febre_amarela_por_mes(FEBRE_AMARELA, "2018/2019")["casos"].sum() == 1


def test_municipios_febre_amarela():
    lista = painel.municipios_febre_amarela(FEBRE_AMARELA, "2016/2017", MUNICIPIOS_FA)
    assert lista.values.tolist() == [["Belo Horizonte", "MG", 8, 3], ["Colatina", "ES", 2, 0]]


def test_graficos_da_febre_amarela():
    periodos = painel.periodos_de_monitoramento("2016/2017", "2018/2019")
    (barras,) = graficos.grafico_periodos(
        painel.febre_amarela_por_periodo(FEBRE_AMARELA, periodos), "2018/2019", tema="dark"
    ).data
    cores = graficos.PALETAS["dark"]
    assert list(barras.x) == ["2016/17", "2017/18", "2018/19"]
    assert list(barras.marker.color) == [cores.referencia, cores.referencia, cores.foco]
    meses = graficos.grafico_meses(painel.febre_amarela_por_mes(FEBRE_AMARELA, "2016/2017"))
    assert meses.layout.yaxis.dtick == 1  # até 10 casos: marcas de 1 em 1, sem frações
    serie = serie_sintetica({2024: 400})
    assert graficos.grafico_anual(painel.totais_anuais(serie), 2024).layout.yaxis.dtick is None
