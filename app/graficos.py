"""Gráficos Plotly do dashboard de arboviroses.

Cores do tema de natureza (`.streamlit/config.toml`): o verde identifica o que está em
foco (o ano escolhido); o histórico fica num marrom-acinzentado discreto. Verde e marrom
de mesma luminosidade se confundem no daltonismo vermelho-verde, então a separação vem
da luminosidade: no tema claro o verde é escuro e o marrom, claro; no escuro, o inverso.
Pares validados com as checagens da skill de visualização (OKLab, simulação de Machado):
CVD 18,5 no claro e 16,2 no escuro, acima do alvo de 8. O histórico fica abaixo de 3:1
contra o fundo claro, como contexto; legenda e tabela carregam a identificação.
Separadores brasileiros (vírgula decimal, ponto de milhar) em eixos e dicas.
"""

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class Paleta:
    """Cores dos gráficos num modo do tema."""

    foco: str  # série em foco
    referencia: str  # mediana histórica e barras fora de foco
    faixa: str  # intervalo interquartil, um véu sobre o fundo
    incompletas: str  # sombra das semanas incompletas
    rotulo: str  # texto discreto dentro do gráfico


PALETAS = {
    "light": Paleta(
        foco="#2d6a35",
        referencia="#a8977c",
        faixa="rgba(168, 151, 124, 0.25)",
        incompletas="rgba(107, 93, 74, 0.10)",
        rotulo="#6b5d4a",
    ),
    "dark": Paleta(
        foco="#5ca965",
        referencia="#6f6452",
        faixa="rgba(111, 100, 82, 0.35)",
        incompletas="rgba(235, 233, 228, 0.07)",
        rotulo="#a89c86",
    ),
}


def paleta(tema: str | None) -> Paleta:
    """Paleta do modo `dark` ou, para qualquer outro valor, do claro."""
    return PALETAS["dark" if tema == "dark" else "light"]


# Fonte de 15 px nos gráficos (o Plotly usa 12), acompanhando o texto das páginas.
_LAYOUT = {
    "separators": ",.",
    "margin": {"l": 8, "r": 8, "t": 44, "b": 8},
    "font": {"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "size": 15},
    "hoverlabel": {"font": {"size": 15}},
}


def _eixo_y(titulo: str, maximo: float) -> dict:
    """Eixo de contagens a partir do zero. Até 10, marcas de 1 em 1: sem isso o Plotly
    usa frações (0,2 caso), que o formato inteiro mostraria como 0, 0, 0, 1."""
    eixo = {"title": titulo, "rangemode": "tozero", "tickformat": ",d"}
    if maximo <= 10:
        eixo["dtick"] = 1
    return eixo


def _periodo(anos: list[int]) -> str:
    return f"{anos[0]}–{anos[-1]}" if len(anos) > 1 else str(anos[0])


def grafico_semanal(
    serie: pd.DataFrame,
    ano: int,
    faixa: pd.DataFrame,
    anos_da_faixa: list[int],
    tema: str | None = "light",
) -> go.Figure:
    """Casos por semana no ano escolhido contra a faixa típica dos anos anteriores.

    A faixa é o intervalo entre o 1º e o 3º quartil de cada semana (metade dos anos
    anteriores fica dentro dela) e a linha cinza, a mediana. As semanas incompletas do
    ano ficam sombreadas.
    """
    cores = paleta(tema)
    figura = go.Figure()
    if not faixa.empty:
        periodo = _periodo(anos_da_faixa)
        figura.add_scatter(
            x=faixa["semana_epi"],
            y=faixa["p75"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
        figura.add_scatter(
            x=faixa["semana_epi"],
            y=faixa["p25"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=cores.faixa,
            name=f"Faixa típica {periodo}",
            hoverinfo="skip",
        )
        figura.add_scatter(
            x=faixa["semana_epi"],
            y=faixa["mediana"],
            mode="lines",
            line={"color": cores.referencia, "width": 2},
            name=f"Mediana {periodo}",
            hovertemplate="Mediana: %{y:,.0f}<extra></extra>",
        )

    do_ano = serie[serie["ano_epi"] == ano]
    inicio = pd.to_datetime(do_ano["inicio_semana"]).dt.strftime("%d/%m/%Y")
    figura.add_scatter(
        x=do_ano["semana_epi"],
        y=do_ano["casos_provaveis"],
        customdata=inicio,
        mode="lines",
        line={"color": cores.foco, "width": 2, "shape": "linear"},
        name=str(ano),
        hovertemplate=f"{ano}: %{{y:,d}} casos (semana de %{{customdata}})<extra></extra>",
    )

    incompletas = do_ano[do_ano["semana_incompleta"]]
    if not incompletas.empty:
        figura.add_vrect(
            x0=incompletas["semana_epi"].min() - 0.5,
            x1=incompletas["semana_epi"].max() + 0.5,
            fillcolor=cores.incompletas,
            line_width=0,
            layer="below",
            annotation_text="incompletas",
            annotation_position="top left",
            annotation_font={"color": cores.rotulo, "size": 14},
        )

    maximos = [do_ano["casos_provaveis"].max(), faixa["p75"].max() if not faixa.empty else 0]
    maximo = max((0 if pd.isna(v) else v) for v in maximos)
    figura.update_layout(
        **_LAYOUT,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={
            "title": "Semana epidemiológica de início dos sintomas",
            "range": [0.5, 53.5],
            "dtick": 4,
            "tick0": 1,
            "showgrid": False,
        },
        yaxis=_eixo_y("Casos prováveis", maximo),
    )
    return figura


def grafico_anual(totais: pd.DataFrame, ano: int, tema: str | None = "light") -> go.Figure:
    """Casos prováveis por ano, com o ano escolhido em destaque."""
    cores = paleta(tema)
    anos = totais["ano_epi"].astype(int).tolist()
    em_andamento = [" (ano em andamento)" if i else "" for i in totais["incompleto"]]
    figura = go.Figure(
        go.Bar(
            x=[str(a) for a in anos],
            y=totais["casos_provaveis"],
            customdata=em_andamento,
            marker={
                "color": [cores.foco if a == ano else cores.referencia for a in anos],
                "cornerradius": 4,
            },
            hovertemplate="%{x}: %{y:,d} casos prováveis%{customdata}<extra></extra>",
        )
    )
    figura.update_layout(
        **_LAYOUT,
        bargap=0.45,
        showlegend=False,
        xaxis={"type": "category", "showgrid": False},
        yaxis=_eixo_y("Casos prováveis", totais["casos_provaveis"].max()),
    )
    return figura


def grafico_periodos(
    por_periodo: pd.DataFrame, periodo: str, tema: str | None = "light"
) -> go.Figure:
    """Casos confirmados de febre amarela por período de monitoramento, com o escolhido
    em destaque."""
    cores = paleta(tema)
    rotulos = [p[:5] + p[-2:] for p in por_periodo["periodo"]]  # 2024/2025 → 2024/25
    figura = go.Figure(
        go.Bar(
            x=rotulos,
            y=por_periodo["casos"],
            customdata=por_periodo["obitos"],
            marker={
                "color": [
                    cores.foco if p == periodo else cores.referencia for p in por_periodo["periodo"]
                ],
                "cornerradius": 4,
            },
            hovertemplate="%{x}: %{y:,d} casos, %{customdata:,d} óbitos<extra></extra>",
        )
    )
    figura.update_layout(
        **_LAYOUT,
        bargap=0.35,
        showlegend=False,
        xaxis={"type": "category", "showgrid": False, "tickangle": -45},
        yaxis=_eixo_y("Casos confirmados", por_periodo["casos"].max()),
    )
    return figura


def grafico_meses(por_mes: pd.DataFrame, tema: str | None = "light") -> go.Figure:
    """Casos confirmados por mês de início dos sintomas no período, de julho a junho."""
    cores = paleta(tema)
    figura = go.Figure(
        go.Bar(
            x=por_mes["rotulo"],
            y=por_mes["casos"],
            customdata=por_mes["obitos"],
            marker={"color": cores.foco, "cornerradius": 4},
            hovertemplate="%{x}: %{y:,d} casos, %{customdata:,d} óbitos<extra></extra>",
        )
    )
    figura.update_layout(
        **_LAYOUT,
        bargap=0.45,
        showlegend=False,
        xaxis={"type": "category", "showgrid": False},
        yaxis=_eixo_y("Casos confirmados", por_mes["casos"].max()),
    )
    return figura
