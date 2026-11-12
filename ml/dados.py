"""Séries semanais de casos prováveis para o treino, lidas do Hub direto para a memória.

O modelo cobre o Brasil, as 27 UFs e os municípios com volume suficiente para que uma
estimativa de tendência faça sentido (AGENTS.md, seção 7.2). Todas as séries de uma
doença dividem o mesmo calendário, porque o ETL completa com zero as semanas sem casos.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from common.config import get_settings

DOENCAS = ("dengue", "chikungunya")
BRASIL = "BR"

# Um município entra no modelo se teve, em média, ao menos este número de casos prováveis
# por semana nas últimas semanas completas (5 anos). Abaixo disso a série é quase só zero
# e um, e o intervalo da estimativa não informa nada.
MEDIA_MINIMA = 5.0
SEMANAS_DE_ELEGIBILIDADE = 260

COLUNAS_CALENDARIO = ["ano_epi", "semana_epi", "inicio_semana", "semana_incompleta"]


@dataclass(frozen=True)
class SeriesSemanais:
    """Séries de uma doença: uma linha por semana publicada, uma coluna por local."""

    doenca: str
    calendario: pd.DataFrame  # ano_epi, semana_epi, inicio_semana, semana_incompleta
    casos: pd.DataFrame  # mesma ordem do calendário; colunas BR, UF (2 dígitos) e município

    @property
    def semanas_completas(self) -> int:
        """Quantidade de semanas completas; as incompletas ficam todas no fim."""
        return int((~self.calendario["semana_incompleta"]).sum())

    @property
    def completas(self) -> pd.DataFrame:
        """Casos só das semanas completas, as únicas que o modelo vê."""
        return self.casos.iloc[: self.semanas_completas]


def nivel_do_local(local: str) -> str:
    """`br`, `uf` (código de 2 dígitos) ou `municipio` (código IBGE de 6 dígitos)."""
    if local == BRASIL:
        return "br"
    return "uf" if len(local) == 2 else "municipio"


def para_painel(tabela: pd.DataFrame, coluna_local: str) -> pd.DataFrame:
    """Tabela longa (uma linha por local e semana) → semanas nas linhas, locais nas colunas."""
    painel = tabela.pivot(
        index=["ano_epi", "semana_epi"], columns=coluna_local, values="casos_provaveis"
    )
    painel.columns.name = None
    return painel.sort_index().reset_index(drop=True)


def elegiveis(completas: pd.DataFrame) -> list[str]:
    """Locais com média de ao menos `MEDIA_MINIMA` casos nas últimas semanas completas."""
    media = completas.tail(SEMANAS_DE_ELEGIBILIDADE).mean()
    return media[media >= MEDIA_MINIMA].index.tolist()


def estender_calendario(calendario: pd.DataFrame, semanas: int) -> pd.DataFrame:
    """Calendário com mais `semanas` depois da última, para datar as estimativas futuras."""
    # Importado aqui: o calendário epidemiológico mora no ETL, e a aplicação não o instala.
    from etl.transform.calendario import semana_epidemiologica

    ultimo_inicio = pd.Timestamp(calendario["inicio_semana"].iloc[-1])
    novas = []
    for k in range(1, semanas + 1):
        inicio = ultimo_inicio + timedelta(weeks=k)
        ano, semana = semana_epidemiologica(inicio.date())
        novas.append({"ano_epi": ano, "semana_epi": semana, "inicio_semana": inicio})
    base = calendario[["ano_epi", "semana_epi", "inicio_semana"]]
    return pd.concat([base, pd.DataFrame(novas)], ignore_index=True)


def _url(caminho: str) -> str:
    return f"hf://datasets/{get_settings().hf_dataset_repo}/{caminho}"


def _municipios_elegiveis(doenca: str, uf: str, completas: int) -> pd.DataFrame:
    tabela = pd.read_parquet(
        _url(f"arboviroses/doenca={doenca}/uf={uf}/casos.parquet"),
        columns=["codigo_ibge6", "ano_epi", "semana_epi", "casos_provaveis"],
    )
    painel = para_painel(tabela, "codigo_ibge6")
    return painel[elegiveis(painel.iloc[:completas])]


def ler_series(doenca: str) -> SeriesSemanais:
    """Brasil, UFs e municípios elegíveis de uma doença, do dataset publicado pelo ETL."""
    resumo = pd.read_parquet(_url("arboviroses/resumo.parquet"), filters=[("doenca", "==", doenca)])
    calendario = (
        resumo.loc[resumo["local"] == BRASIL, COLUNAS_CALENDARIO]
        .sort_values(["ano_epi", "semana_epi"])
        .reset_index(drop=True)
    )
    calendario["inicio_semana"] = pd.to_datetime(calendario["inicio_semana"])
    completas = int((~calendario["semana_incompleta"]).sum())
    if calendario["semana_incompleta"].iloc[:completas].any():
        raise ValueError(f"{doenca}: há semana incompleta no meio da série.")

    por_uf = para_painel(resumo, "local")
    ufs = [local for local in por_uf.columns if local != BRASIL]
    with ThreadPoolExecutor(max_workers=4) as executor:
        municipios = list(
            executor.map(lambda uf: _municipios_elegiveis(doenca, uf, completas), ufs)
        )
    casos = pd.concat([por_uf[[BRASIL, *ufs]], *municipios], axis=1)
    if len(casos) != len(calendario) or casos.isna().any().any():
        raise ValueError(f"{doenca}: as séries não dividem o mesmo calendário.")
    return SeriesSemanais(doenca=doenca, calendario=calendario, casos=casos)
