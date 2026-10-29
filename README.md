# Radar Epidemiológico

Aplicação em Streamlit para vigilância de dengue, zika, chikungunya e febre amarela a
partir dos dados abertos do Ministério da Saúde: dashboard histórico, estimativa de
tendência, mapa das UBS mais próximas por CEP e chatbot RAG sobre os manuais oficiais.

Projeto de extensão universitária. Contexto, requisitos e decisões técnicas estão em
[`AGENTS.md`](AGENTS.md); a apresentação e o diagrama de arquitetura, em [`docs/`](docs/).

## Arquitetura

```
Fontes (DEMAS, IBGE, BrasilAPI, manuais MS)
  → GitHub Actions: ETL, treino, indexação
  → Hugging Face Hub: Parquet, modelo, índice FAISS
  → Streamlit Community Cloud
```

Nenhum dado é persistido localmente nem em serviço pago: o pipeline roda no CI e publica
no Hub; a aplicação lê do Hub no startup.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `notebooks/` | Etapa 0: análise exploratória qualitativa e quantitativa |
| `common/` | Configuração compartilhada (`pydantic-settings`) |
| `etl/` | Extração (DEMAS, IBGE), normalização, validação Pandera e publicação no Hub |
| `ml/` | Atributos, treino e backtest do modelo de tendência |
| `rag/` | Ingestão dos manuais, chunking, indexação, busca e filtro de sinais de alarme |
| `app/` | Aplicação Streamlit, suas páginas e o `requirements.txt` do deploy |
| `.github/workflows/` | CI, ETL, treino e indexação |
| `tests/` | Testes `pytest` |

## Ambiente de desenvolvimento

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # instala dev, etl, ml, app e eda
uv run pytest                # testes
uv run ruff check . && uv run ruff format --check .
uv run jupyter lab notebooks/
uv run streamlit run app/main.py
```

Os grupos `rag` e `indexacao` puxam torch e Docling e ficam fora do padrão:
`uv sync --group rag --group indexacao`.

Copie `.env.example` para `.env` e preencha os segredos. No GitHub, cadastre
`RADAR_HF_TOKEN` em *Secrets* e `RADAR_HF_DATASET_REPO` em *Variables*.

## Deploy

A aplicação roda no [Streamlit Community Cloud](https://share.streamlit.io), ligada a
este repositório: cada push na `main` atualiza o app. Configuração do app:

- *Main file path*: `app/main.py`; *Python version* (em *Advanced settings*): 3.12.
- Dependências: `app/requirements.txt`, gerado a partir do `uv.lock`. Ao mudar os grupos
  `app` ou `ml`, regenere com
  `uv export --no-hashes --no-dev --no-default-groups --group app --group ml --no-emit-project -o app/requirements.txt`.
  O CI falha se o arquivo estiver desatualizado.
- Segredos em *Settings > Secrets*, em TOML, com os mesmos nomes do `.env`.

## Ordem de implementação

0. Análise exploratória dos dados (`notebooks/01_analise_exploratoria.ipynb`).
1. Cliente da DEMAS com paginação e retry.
2. ETL de UBS e publicação no Hub.
3. Mapa de UBS.
4. ETL de arboviroses e dicionário de municípios do IBGE.
5. Dashboard histórico.
6. Previsão, a partir do baseline sazonal.
7. Chatbot RAG.
