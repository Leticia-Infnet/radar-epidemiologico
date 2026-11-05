"""Cálculos do dashboard de arboviroses (AGENTS.md, seção 7.1), sem Streamlit.

Dengue, chikungunya e zika vêm do ETL `arboviroses`: casos prováveis e óbitos por semana
epidemiológica de início dos sintomas e município de residência, com zero nas semanas
sem casos. A febre amarela vem do ETL `febre_amarela`: casos confirmados por município
provável de infecção, período de monitoramento (julho a junho) e mês.
"""

from dataclasses import dataclass

import pandas as pd

DOENCAS = {
    "dengue": "Dengue",
    "chikungunya": "Chikungunya",
    "zika": "Zika",
    "febre_amarela": "Febre amarela",
}
FEBRE_AMARELA = "febre_amarela"
BRASIL = "BR"

# Código IBGE da UF → (sigla, nome). As 27 UFs não mudam; os municípios vêm do Hub.
UFS = {
    "11": ("RO", "Rondônia"),
    "12": ("AC", "Acre"),
    "13": ("AM", "Amazonas"),
    "14": ("RR", "Roraima"),
    "15": ("PA", "Pará"),
    "16": ("AP", "Amapá"),
    "17": ("TO", "Tocantins"),
    "21": ("MA", "Maranhão"),
    "22": ("PI", "Piauí"),
    "23": ("CE", "Ceará"),
    "24": ("RN", "Rio Grande do Norte"),
    "25": ("PB", "Paraíba"),
    "26": ("PE", "Pernambuco"),
    "27": ("AL", "Alagoas"),
    "28": ("SE", "Sergipe"),
    "29": ("BA", "Bahia"),
    "31": ("MG", "Minas Gerais"),
    "32": ("ES", "Espírito Santo"),
    "33": ("RJ", "Rio de Janeiro"),
    "35": ("SP", "São Paulo"),
    "41": ("PR", "Paraná"),
    "42": ("SC", "Santa Catarina"),
    "43": ("RS", "Rio Grande do Sul"),
    "50": ("MS", "Mato Grosso do Sul"),
    "51": ("MT", "Mato Grosso"),
    "52": ("GO", "Goiás"),
    "53": ("DF", "Distrito Federal"),
}

# Anos anteriores ao escolhido que formam a faixa histórica (mediana e quartis).
ANOS_DE_REFERENCIA = 10
# Abaixo disto no ano anterior, a variação percentual oscila demais para informar algo
# (3 casos contra 1 seria "+200%").
MINIMO_PARA_VARIACAO = 20

COLUNAS_SERIE = [
    "ano_epi",
    "semana_epi",
    "inicio_semana",
    "casos_provaveis",
    "obitos",
    "semana_incompleta",
]


def serie_do_local(
    resumo: pd.DataFrame, casos_uf: pd.DataFrame | None, doenca: str, local: str
) -> pd.DataFrame:
    """Série semanal de um local: `BR`, código de UF (2 dígitos) ou de município (6).

    Brasil e UF vêm do resumo; município, da partição da UF (`casos_uf`).
    """
    if len(local) == 6:
        if casos_uf is None:
            raise ValueError("A série de um município precisa da partição da UF.")
        linhas = casos_uf[casos_uf["codigo_ibge6"] == local]
    else:
        linhas = resumo[(resumo["doenca"] == doenca) & (resumo["local"] == local)]
    return linhas[COLUNAS_SERIE].sort_values(["ano_epi", "semana_epi"]).reset_index(drop=True)


def populacao_do_local(populacao: pd.DataFrame, local: str) -> int | None:
    """Soma da população dos municípios do local; None se nenhum tiver população."""
    codigos = populacao["codigo_ibge6"]
    if local == BRASIL:
        selecao = populacao
    elif len(local) == 2:
        selecao = populacao[codigos.str.startswith(local)]
    else:
        selecao = populacao[codigos == local]
    total = int(selecao["populacao"].sum())
    return total or None


@dataclass(frozen=True)
class Indicadores:
    """Números de destaque do ano escolhido."""

    casos: int
    obitos: int
    incidencia: float | None  # casos por 100 mil habitantes
    variacao: float | None  # fração em relação ao mesmo período do ano anterior
    semana_comparada: int | None  # última semana completa usada na comparação
    semana_pico: int | None
    casos_no_pico: int
    tem_semanas_incompletas: bool


def indicadores(serie: pd.DataFrame, ano: int, populacao: int | None) -> Indicadores:
    """Casos, óbitos, incidência, variação e pico do ano na série do local.

    A variação compara o ano escolhido com o anterior até a mesma semana, usando só
    semanas completas: as incompletas subestimariam o ano corrente. Fica vazia quando o
    ano anterior tem menos de `MINIMO_PARA_VARIACAO` casos no período.
    """
    do_ano = serie[serie["ano_epi"] == ano]
    casos = int(do_ano["casos_provaveis"].sum())
    completas = do_ano[~do_ano["semana_incompleta"]]
    semana_comparada = int(completas["semana_epi"].max()) if not completas.empty else None

    variacao = None
    if semana_comparada is not None:
        anterior = serie[(serie["ano_epi"] == ano - 1) & (serie["semana_epi"] <= semana_comparada)]
        base = int(anterior["casos_provaveis"].sum())
        if not anterior.empty and base >= MINIMO_PARA_VARIACAO:
            atual = int(completas["casos_provaveis"].sum())
            variacao = atual / base - 1

    semana_pico, casos_no_pico = None, 0
    if casos > 0:
        pico = do_ano.loc[do_ano["casos_provaveis"].idxmax()]
        semana_pico, casos_no_pico = int(pico["semana_epi"]), int(pico["casos_provaveis"])

    return Indicadores(
        casos=casos,
        obitos=int(do_ano["obitos"].sum()),
        incidencia=casos / populacao * 100_000 if populacao else None,
        variacao=variacao,
        semana_comparada=semana_comparada,
        semana_pico=semana_pico,
        casos_no_pico=casos_no_pico,
        tem_semanas_incompletas=bool(do_ano["semana_incompleta"].any()),
    )


def faixa_historica(
    serie: pd.DataFrame, ano: int, anos: int = ANOS_DE_REFERENCIA
) -> tuple[pd.DataFrame, list[int]]:
    """Mediana e quartis por semana nos `anos` anteriores ao escolhido.

    Devolve a tabela (`semana_epi`, `mediana`, `p25`, `p75`) e os anos usados; vazia se
    não houver ano anterior na série.
    """
    anteriores = serie[(serie["ano_epi"] < ano) & (serie["ano_epi"] >= ano - anos)]
    usados = sorted(anteriores["ano_epi"].unique().tolist())
    if not usados:
        return pd.DataFrame(columns=["semana_epi", "mediana", "p25", "p75"]), []
    por_semana = anteriores.groupby("semana_epi")["casos_provaveis"]
    faixa = pd.DataFrame(
        {
            "mediana": por_semana.median(),
            "p25": por_semana.quantile(0.25),
            "p75": por_semana.quantile(0.75),
        }
    ).reset_index()
    return faixa, usados


def totais_anuais(serie: pd.DataFrame) -> pd.DataFrame:
    """Casos e óbitos por ano, com a marca dos anos que têm semanas incompletas."""
    return (
        serie.groupby("ano_epi")
        .agg(
            casos_provaveis=("casos_provaveis", "sum"),
            obitos=("obitos", "sum"),
            incompleto=("semana_incompleta", "any"),
        )
        .reset_index()
    )


def ranking_municipios(
    casos_uf: pd.DataFrame, ano: int, municipios: pd.DataFrame, populacao: pd.DataFrame
) -> pd.DataFrame:
    """Municípios da UF no ano, do maior para o menor coeficiente de incidência.

    Municípios sem população ficam no fim, sem incidência.
    """
    do_ano = casos_uf[casos_uf["ano_epi"] == ano]
    totais = (
        do_ano.groupby("codigo_ibge6", observed=True)
        .agg(casos=("casos_provaveis", "sum"), obitos=("obitos", "sum"))
        .reset_index()
    )
    totais["codigo_ibge6"] = totais["codigo_ibge6"].astype(str)
    tabela = totais.merge(
        municipios[["codigo_ibge6", "nome"]], on="codigo_ibge6", how="left"
    ).merge(populacao[["codigo_ibge6", "populacao"]], on="codigo_ibge6", how="left")
    tabela["incidencia"] = tabela["casos"] / tabela["populacao"] * 100_000
    return (
        tabela.sort_values(["incidencia", "casos"], ascending=False, na_position="last")
        .reset_index(drop=True)[["nome", "casos", "incidencia", "obitos", "populacao"]]
        .rename(columns={"nome": "municipio"})
    )


def formatar_inteiro(valor: float) -> str:
    """`6564924` → `6.564.924`."""
    return f"{round(valor):,}".replace(",", ".")


def formatar_decimal(valor: float, casas: int = 1) -> str:
    """`1234.56` → `1.234,6`."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def formatar_variacao(valor: float) -> str:
    """`-0.7064` → `-70,6%`; `0.25` → `+25,0%`."""
    sinal = "+" if valor > 0 else ""
    return f"{sinal}{formatar_decimal(valor * 100)}%"


# Meses de um período de monitoramento da febre amarela, de julho a junho.
MESES_DO_PERIODO = (7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6)
NOMES_DOS_MESES = (
    "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"
)  # fmt: skip


def febre_amarela_do_local(tabela: pd.DataFrame, local: str) -> pd.DataFrame:
    """Linhas da febre amarela de um local: `BR`, UF (2 dígitos) ou município (6)."""
    if local == BRASIL:
        return tabela
    coluna = "uf_codigo" if len(local) == 2 else "codigo_ibge6"
    return tabela[tabela[coluna] == local]


def periodos_de_monitoramento(primeiro: str, ultimo: str) -> list[str]:
    """Todos os períodos entre dois, inclusive os sem casos: `2010/2011` a `2012/2013`."""
    inicio, fim = int(primeiro[:4]), int(ultimo[:4])
    return [f"{ano}/{ano + 1}" for ano in range(inicio, fim + 1)]


@dataclass(frozen=True)
class IndicadoresFebreAmarela:
    """Números de destaque de um período de monitoramento."""

    casos: int
    obitos: int
    letalidade: float | None  # fração de óbitos entre os casos
    municipios: int  # municípios prováveis de infecção com casos


def indicadores_febre_amarela(dados: pd.DataFrame, periodo: str) -> IndicadoresFebreAmarela:
    """Casos, óbitos, letalidade e municípios com casos no período."""
    do_periodo = dados[dados["periodo"] == periodo]
    casos, obitos = int(do_periodo["casos"].sum()), int(do_periodo["obitos"].sum())
    return IndicadoresFebreAmarela(
        casos=casos,
        obitos=obitos,
        letalidade=obitos / casos if casos else None,
        municipios=int(do_periodo["codigo_ibge6"].dropna().nunique()),
    )


def febre_amarela_por_periodo(dados: pd.DataFrame, periodos: list[str]) -> pd.DataFrame:
    """Casos e óbitos de cada período, com zero nos períodos sem casos."""
    soma = dados.groupby("periodo")[["casos", "obitos"]].sum()
    return soma.reindex(periodos, fill_value=0).rename_axis("periodo").reset_index()


def febre_amarela_por_mes(dados: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Casos e óbitos por mês do período, de julho a junho, com zero nos meses sem casos.

    Casos sem mês de início dos sintomas ficam de fora (11 de 2.880 em 2026-11).
    """
    inicio = int(periodo[:4])
    do_periodo = dados[(dados["periodo"] == periodo) & dados["mes"].notna()]
    soma = do_periodo.groupby(do_periodo["mes"].astype(int))[["casos", "obitos"]].sum()
    tabela = soma.reindex(list(MESES_DO_PERIODO), fill_value=0).rename_axis("mes").reset_index()
    tabela["rotulo"] = [
        f"{NOMES_DOS_MESES[m - 1]}/{(inicio if m >= 7 else inicio + 1) % 100:02d}"
        for m in tabela["mes"]
    ]
    return tabela


def municipios_febre_amarela(
    dados: pd.DataFrame, periodo: str, municipios: pd.DataFrame
) -> pd.DataFrame:
    """Municípios prováveis de infecção do período, do maior para o menor número de casos."""
    do_periodo = dados[(dados["periodo"] == periodo) & dados["codigo_ibge6"].notna()]
    totais = do_periodo.groupby("codigo_ibge6")[["casos", "obitos"]].sum().reset_index()
    tabela = totais.merge(
        municipios[["codigo_ibge6", "nome", "uf_sigla"]], on="codigo_ibge6", how="left"
    )
    tabela["nome"] = tabela["nome"].fillna("Município " + tabela["codigo_ibge6"])
    return (
        tabela.sort_values(["casos", "obitos", "nome"], ascending=[False, False, True])
        .reset_index(drop=True)[["nome", "uf_sigla", "casos", "obitos"]]
        .rename(columns={"nome": "municipio", "uf_sigla": "uf"})
    )
