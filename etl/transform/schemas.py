"""Contratos Pandera das tabelas publicadas no Hub."""

import pandera.polars as pa

from etl.transform.normalize import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


class UBS(pa.DataFrameModel):
    """Cadastro de UBS pronto para a busca por proximidade."""

    cnes: str = pa.Field(unique=True, str_matches=r"^\d{7}$")
    nome: str
    codigo_ibge6: str = pa.Field(str_matches=r"^\d{6}$")
    uf_codigo: str = pa.Field(str_matches=r"^\d{2}$")
    logradouro: str = pa.Field(nullable=True)
    bairro: str = pa.Field(nullable=True)
    latitude: float = pa.Field(ge=LAT_MIN, le=LAT_MAX)
    longitude: float = pa.Field(ge=LON_MIN, le=LON_MAX)
    # Até 2 casas decimais (precisão de cerca de 1 km).
    coordenada_aproximada: bool


class Municipio(pa.DataFrameModel):
    """Dicionário código IBGE ↔ nome."""

    codigo_ibge7: str = pa.Field(unique=True, str_matches=r"^\d{7}$")
    codigo_ibge6: str = pa.Field(unique=True, str_matches=r"^\d{6}$")
    nome: str
    uf_sigla: str = pa.Field(str_length=2)
    uf_codigo: str
    regiao: str


class NotificacoesSemanais(pa.DataFrameModel):
    """Notificações agregadas por doença, município e semana epidemiológica."""

    doenca: str = pa.Field(isin=["dengue", "zika", "chikungunya", "febre_amarela"])
    codigo_ibge6: str = pa.Field(str_matches=r"^\d{6}$")
    ano_epi: int = pa.Field(ge=2000)
    semana_epi: int = pa.Field(ge=1, le=53)
    notificacoes: int = pa.Field(ge=0)
    casos_provaveis: int = pa.Field(ge=0)
