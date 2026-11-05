"""Contratos Pandera das tabelas publicadas no Hub."""

from datetime import date

import pandera.polars as pa

from etl.extract.datasus import AGRAVOS_SINAN as DOENCAS
from etl.transform.aggregate import BRASIL, UF_CODIGOS
from etl.transform.cbo import GRUPOS
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


class ProfissionalUBS(pa.DataFrameModel):
    """Profissionais de saúde distintos por UBS e ocupação (CBO), sem dados pessoais."""

    cnes: str = pa.Field(str_matches=r"^\d{7}$")
    grupo: str = pa.Field(isin=list(GRUPOS))
    cbo: str = pa.Field(str_length=6)
    ocupacao: str
    profissionais: int = pa.Field(ge=1)
    competencia: str = pa.Field(str_matches=r"^\d{4}-\d{2}$")

    class Config:
        unique = ["cnes", "cbo"]


class Municipio(pa.DataFrameModel):
    """Dicionário código IBGE ↔ nome."""

    codigo_ibge7: str = pa.Field(unique=True, str_matches=r"^\d{7}$")
    codigo_ibge6: str = pa.Field(unique=True, str_matches=r"^\d{6}$")
    nome: str
    uf_sigla: str = pa.Field(str_length=2)
    uf_codigo: str
    regiao: str


class CasosSemanais(pa.DataFrameModel):
    """Casos prováveis e óbitos por doença, município de residência e semana de início dos
    sintomas, com zero nas semanas sem casos."""

    doenca: str = pa.Field(isin=list(DOENCAS))
    codigo_ibge6: str = pa.Field(str_matches=r"^\d{6}$")
    uf_codigo: str = pa.Field(isin=sorted(UF_CODIGOS))
    ano_epi: int = pa.Field(ge=2000)
    semana_epi: int = pa.Field(ge=1, le=53)
    inicio_semana: date
    casos_provaveis: int = pa.Field(ge=0)
    obitos: int = pa.Field(ge=0)
    semana_incompleta: bool

    class Config:
        unique = ["doenca", "codigo_ibge6", "ano_epi", "semana_epi"]


class ResumoSemanal(pa.DataFrameModel):
    """As mesmas séries por UF (código IBGE de 2 dígitos) e para o Brasil (`BR`)."""

    doenca: str = pa.Field(isin=list(DOENCAS))
    local: str = pa.Field(isin=[BRASIL, *sorted(UF_CODIGOS)])
    ano_epi: int = pa.Field(ge=2000)
    semana_epi: int = pa.Field(ge=1, le=53)
    inicio_semana: date
    casos_provaveis: int = pa.Field(ge=0)
    obitos: int = pa.Field(ge=0)
    semana_incompleta: bool

    class Config:
        unique = ["doenca", "local", "ano_epi", "semana_epi"]


class FebreAmarelaMensal(pa.DataFrameModel):
    """Casos humanos confirmados de febre amarela por período de monitoramento (julho a
    junho), mês de início dos sintomas e município provável de infecção."""

    periodo: str = pa.Field(str_matches=r"^\d{4}/\d{4}$")
    ano_inicio_periodo: int = pa.Field(ge=1990)
    ano: int = pa.Field(nullable=True, ge=1990)
    mes: int = pa.Field(nullable=True, ge=1, le=12)
    codigo_ibge6: str = pa.Field(nullable=True, str_matches=r"^\d{6}$")
    uf_codigo: str = pa.Field(nullable=True, isin=sorted(UF_CODIGOS))
    casos: int = pa.Field(ge=1)
    obitos: int = pa.Field(ge=0)


class Populacao(pa.DataFrameModel):
    """População estimada por município, usada no cálculo de incidência."""

    codigo_ibge6: str = pa.Field(unique=True, str_matches=r"^\d{6}$")
    populacao: int = pa.Field(gt=0)
    ano_referencia: int = pa.Field(ge=2000)
