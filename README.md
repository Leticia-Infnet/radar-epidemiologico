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
| `etl/` | Extração (DEMAS, IBGE, FTP do DATASUS), normalização, validação Pandera e publicação no Hub |
| `ml/` | Séries, atributos, modelos candidatos, backtest e treino da estimativa de tendência |
| `rag/` | Ingestão dos manuais, chunking, indexação, busca e filtro de sinais de alarme |
| `app/` | Aplicação Streamlit, suas páginas e o `requirements.txt` do deploy |
| `.github/workflows/` | CI, ETL, treino e indexação |
| `tests/` | Testes `pytest` |

## Ambiente de desenvolvimento

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # instala dev, etl, ml, app, rag e eda
uv run pytest                # testes
uv run ruff check . && uv run ruff format --check .
uv run jupyter lab notebooks/
uv run streamlit run app/main.py
```

O grupo `indexacao` puxa torch e Docling e fica fora do padrão:
`uv sync --group indexacao`.

Copie `.env.example` para `.env` e preencha os segredos. No GitHub, cadastre
`RADAR_HF_TOKEN` em *Secrets* e `RADAR_HF_DATASET_REPO` em *Variables*.

## ETL

```bash
uv run python -m etl.pipeline municipios ubs --sem-publicar   # extrai e valida
uv run pytest -m integracao                                   # testes contra as APIs reais
```

A publicação no Hub é feita pelo workflow `etl` (*Actions → etl → Run workflow*), que
envia cada tabela para `<RADAR_HF_DATASET_REPO>/<tabela>/` com um `metadados.json`.

O dicionário de municípios é a exceção: a API do IBGE recusa conexões vindas do GitHub
Actions, então essa etapa roda de uma máquina no Brasil, com `RADAR_HF_TOKEN` (escrita)
no `.env`. Só é preciso repetir quando o IBGE criar ou alterar municípios:

```bash
uv run python -m etl.pipeline municipios
```

A etapa `profissionais` conta os profissionais de saúde de cada UBS por ocupação (CBO),
a partir dos arquivos mensais `PF<UF><AAMM>.dbc` do CNES no FTP do DATASUS. Usa a
competência mais recente já publicada para todas as UFs e as UBS que estão no Hub, então
deve rodar depois de `ubs`. Baixa cerca de 350 MB, uma UF por vez, e publica só as
contagens, sem nomes nem documentos dos profissionais.

A etapa `arboviroses` agrega dengue, chikungunya e zika dos arquivos nacionais do SINAN no
mesmo FTP: casos prováveis e óbitos por município de residência e semana epidemiológica de
início dos sintomas, do Brasil inteiro (dengue desde 2010, chikungunya desde 2014, zika
desde 2015). Usa o dicionário de municípios do Hub. A etapa `populacao` publica a
população de 2022 por município, para o cálculo de incidência. A etapa `febre_amarela` publica
os casos humanos confirmados da DEMAS por período de monitoramento (julho a junho), mês e
município provável de infecção. O workflow `etl` roda toda segunda-feira, porque os
arquivos preliminares do ano corrente mudam ao longo do ano.

Um piso de linhas por tabela impede que uma extração vazia ou parcial substitua os dados
publicados.

## Estimativa de tendência

```bash
uv run python -m ml.train --sem-publicar   # treina, avalia e valida, sem enviar ao Hub
```

O treino lê as séries semanais de dengue e chikungunya publicadas pelo ETL (Brasil, UFs e
municípios com média de 5 ou mais casos por semana nos últimos 5 anos), compara os
candidatos num backtest de origem móvel (baseline sazonal, persistência, Ridge com
atributos sazonais e GLM binomial negativa) e publica em `previsao/` as estimativas das
4 semanas incompletas e das 4 seguintes, com intervalos de 50% e 90%, as métricas do
backtest e o modelo. Nada é publicado se o modelo escolhido não superar os baselines.

O workflow `train` roda ao fim de cada `etl` concluído com sucesso e também manualmente
(*Actions → train → Run workflow*). A aplicação só lê as estimativas prontas: o modelo
nunca é treinado nem executado em request. O resultado é uma estimativa estatística de
tendência, não uma previsão oficial.

## Chatbot RAG

Orienta sobre dengue, chikungunya, zika e febre amarela a partir de cinco documentos do
Ministério da Saúde (lista em `rag/fontes.yaml`), **sem diagnosticar**: toda resposta cita
manual e página, perguntas sem resposta nos manuais recebem "não encontrei", e sinais de
alarme da dengue interrompem a conversa com a orientação de procurar atendimento.

```bash
uv run --group indexacao python -m rag.index --sem-publicar   # baixa, converte, indexa e avalia
uv run --group indexacao python -m rag.index                  # e publica em rag/ no Hub
uv run --with pypdf --with pyyaml python -m rag.ingest --sumario <id>      # páginas do PDF
uv run --with pypdf --with pyyaml python -m rag.ingest --procurar <id> "termo"
```

A indexação baixa os PDFs para um diretório temporário, converte com o Docling (OCR só
nos manuais marcados), divide em trechos por seção e gera os embeddings do
`multilingual-e5-small` em ONNX (sem torch, o mesmo modelo no CI e no app). Antes de
publicar em `rag/` no Hub, avalia a busca híbrida (BM25 + densa, reordenada por um
cross-encoder multilíngue em ONNX) com as perguntas de
`rag/avaliacao.yaml`, calibra o limiar de similaridade e passa por um portão de
qualidade.

A indexação roda **localmente**, com `RADAR_HF_TOKEN` (escrita) no `.env`: a Biblioteca
Virtual em Saúde recusa downloads vindos do GitHub Actions (HTTP 403), como a API do IBGE.
Só é preciso reindexar quando mudam os manuais (`rag/fontes.yaml`), os sinônimos
(`rag/sinonimos.yaml`), as perguntas de avaliação ou o chunking.

As respostas vêm do `openai/gpt-oss-120b` no Groq (cota gratuita; reserva no
`gpt-oss-20b`), com chave em `RADAR_LLM_API_KEY`. As perguntas sem resposta vão para um
dataset privado no Hub (`RADAR_HF_CONSULTAS_TOKEN`, token *fine-grained* com escrita só
nele), sem identificação. O workflow manual `avaliacao` mede fidelidade e precisão de
contexto das respostas com o Ragas e gasta a cota do Groq.

## Deploy

A aplicação roda no [Streamlit Community Cloud](https://share.streamlit.io), ligada a
este repositório: cada push na `main` atualiza o app. Configuração do app:

- *Main file path*: `app/main.py`; *Python version* (em *Advanced settings*): 3.12.
- Dependências: `app/requirements.txt`, gerado a partir do `uv.lock`. Ao mudar os grupos
  `app`, `ml` ou `rag`, regenere com
  `uv export --no-hashes --no-dev --no-default-groups --group app --group ml --group rag --no-emit-project -o app/requirements.txt`.
  O CI falha se o arquivo estiver desatualizado.
- Segredos em *Settings > Secrets*, em TOML, com os mesmos nomes do `.env`: o chatbot usa
  `RADAR_LLM_API_KEY` e `RADAR_HF_CONSULTAS_TOKEN`.

## Ordem de implementação

0. Análise exploratória dos dados (`notebooks/01_analise_exploratoria.ipynb`).
1. Cliente da DEMAS com paginação e retry.
2. ETL de UBS e publicação no Hub.
3. Mapa de UBS.
4. ETL de arboviroses e dicionário de municípios do IBGE.
5. Dashboard histórico.
6. Estimativa de tendência, a partir do baseline sazonal.
7. Chatbot RAG.
