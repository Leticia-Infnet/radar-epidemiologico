"""Configuração central do projeto, lida de variáveis de ambiente com prefixo RADAR_."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros compartilhados por ETL, treino, indexação e aplicação."""

    model_config = SettingsConfigDict(env_prefix="RADAR_", env_file=".env", extra="ignore")

    demas_base_url: str = "https://apidadosabertos.saude.gov.br"
    # Verificado em 2026-10-28: o servidor aceita até 1000 registros por requisição.
    demas_page_size: int = 1000
    demas_max_concurrency: int = 4
    demas_timeout_s: float = 120.0

    ibge_municipios_url: str = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
    brasilapi_cep_url: str = "https://brasilapi.com.br/api/cep/v2/{cep}"
    awesomeapi_cep_url: str = "https://cep.awesomeapi.com.br/json/{cep}"
    # Cabeçalhos HTTP aceitam só ASCII.
    user_agent: str = "radar-epidemiologico/0.1 (projeto de extensao universitaria)"

    hf_token: SecretStr | None = None
    hf_dataset_repo: str = "LeticiaF/radar-epidemiologico-dados"
    hf_space_repo: str = "LeticiaF/radar-epidemiologico"

    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


@lru_cache
def get_settings() -> Settings:
    """Devolve a instância única de configuração."""
    return Settings()
