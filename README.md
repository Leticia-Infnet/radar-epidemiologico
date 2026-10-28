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
  → Streamlit no Hugging Face Spaces
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
| `app/` | Aplicação Streamlit e suas páginas |
| `space/` | Metadados do Hugging Face Space |
| `.github/workflows/` | CI, ETL, treino, indexação e deploy do Space |
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
`RADAR_HF_TOKEN` em *Secrets* e `RADAR_HF_DATASET_REPO` e `RADAR_HF_SPACE_REPO` em
*Variables*.

## Ordem de implementação

0. Análise exploratória dos dados (`notebooks/01_analise_exploratoria.ipynb`).
1. Cliente da DEMAS com paginação e retry.
2. ETL de UBS e publicação no Hub.
3. Mapa de UBS.
4. ETL de arboviroses e dicionário de municípios do IBGE.
5. Dashboard histórico.
6. Previsão, a partir do baseline sazonal.
7. Chatbot RAG.
