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

    # FTP do DATASUS: profissionais do CNES (PF<UF><AAMM>.dbc) e tabelas auxiliares.
    datasus_ftp_host: str = "ftp.datasus.gov.br"
    datasus_cnes_pf_dir: str = "/dissemin/publicos/CNES/200508_/Dados/PF"
    datasus_cnes_tabelas: str = "/dissemin/publicos/CNES/200508_/Auxiliar/TAB_CNES.zip"
    datasus_sinan_finais_dir: str = "/dissemin/publicos/SINAN/DADOS/FINAIS"
    datasus_sinan_prelim_dir: str = "/dissemin/publicos/SINAN/DADOS/PRELIM"
    datasus_ftp_connect_timeout_s: float = 30.0
    datasus_ftp_timeout_s: float = 300.0

    ibge_municipios_url: str = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
    brasilapi_cep_url: str = "https://brasilapi.com.br/api/cep/v2/{cep}"
    awesomeapi_cep_url: str = "https://cep.awesomeapi.com.br/json/{cep}"
    viacep_url: str = "https://viacep.com.br/ws/{cep}/json/"
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    # OSRM da FOSSGIS, com perfis a pé (routed-foot) e de carro (routed-car).
    osrm_base_url: str = "https://routing.openstreetmap.de"
    # Cabeçalhos HTTP aceitam só ASCII.
    user_agent: str = "radar-epidemiologico/0.1 (projeto de extensao universitaria)"

    hf_token: SecretStr | None = None
    hf_dataset_repo: str = "LeticiaF/radar-epidemiologico-dados"

    # LLM por API compatível com a da OpenAI (Groq). O Llama 3.3 70B foi desligado no Groq
    # em 2026-08-16; o gpt-oss tem pesos abertos (Apache 2.0). O segundo modelo é a reserva
    # quando a cota gratuita do primeiro acaba.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: SecretStr | None = None
    llm_modelos: list[str] = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    # gpt-oss: menos tokens de raciocínio, que também contam na cota.
    llm_reasoning_effort: str | None = "low"
    llm_timeout_s: float = 60.0

    # Embeddings em ONNX: o mesmo arquivo no CI e no app, sem torch no deploy. A versão
    # quantizada em int8 roda em qualquer CPU x86-64 (apesar do nome) e ocupa 520 MB de RAM
    # contra 885 MB da fp32, com cosseno acima de 0,99 entre as duas (medido em 2026-11-17).
    embedding_repo: str = "intfloat/multilingual-e5-small"
    embedding_arquivo: str = "onnx/model_qint8_avx512_vnni.onnx"
    embedding_tokenizer: str = "onnx/tokenizer.json"

    # Reranker (cross-encoder multilíngue em ONNX int8) sobre os primeiros candidatos da
    # busca híbrida. Medido em 2026-11-17 com 10 candidatos e pares de até 256 tokens:
    # acerto entre os 4 primeiros de 0,81 para 0,92, cerca de 1,5 s e 330 MB a mais.
    # `RADAR_RERANK_ATIVO=false` desliga no índice e no app.
    rerank_ativo: bool = True
    rerank_repo: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_arquivo: str = "onnx/model_qint8_avx512_vnni.onnx"
    rerank_tokenizer: str = "tokenizer.json"
    rerank_candidatos: int = 10
    rerank_max_tokens: int = 256

    # Perguntas sem resposta do chatbot: dataset privado, com token de escrita só nele.
    hf_consultas_repo: str = "LeticiaF/radar-epidemiologico-consultas"
    hf_consultas_token: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Devolve a instância única de configuração."""
    return Settings()
