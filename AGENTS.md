# Projeto: aplicação interativa de arboviroses

Documento de contexto para uso com Claude Code. Contém requisitos, arquitetura, stack
definida e armadilhas já identificadas. Leia antes de gerar ou alterar código.

---

## 1. Objetivo

Aplicação web em Streamlit que permite ao usuário explorar dados de arboviroses no
Brasil, prever tendência de notificações, localizar a unidade básica de saúde mais
próxima a partir do CEP, e conversar com um chatbot RAG alimentado por manuais oficiais
do Ministério da Saúde.

## 2. Restrições inegociáveis

1. **Streamlit** como framework de interface.
2. Apenas software **open source e sem custo**. Nada de licença paga.
3. Os dados **não podem** ser armazenados em plataforma paga **nem localmente** na
   máquina do usuário. Todo estado persistente vive em serviço gratuito na nuvem.

Consequência direta: o pipeline de dados roda em CI na nuvem e grava em repositório
gratuito. Nada de banco local, nada de `data/` versionado com CSVs grandes.

---

## 3. Descoberta crítica sobre a API DEMAS

> **Verificação de 2026-10-28 contra a API real** (detalhes em
> `notebooks/01_analise_exploratoria.ipynb` e `etl/extract/demas.py`). Ela corrige o texto
> abaixo para os endpoints usados pelo projeto:
>
> - **Nenhum exige autenticação**: arboviroses, febre amarela e UBS respondem HTTP 200
>   sem token. O ETL não precisa de passo de login.
> - **`limit` vai até 1000**, não 20 (o Swagger em `/static/swagger.json` diz "menor ou
>   igual 1000"; acima disso o servidor trunca para 1000 sem erro).
> - **`offset` desloca por registro**, embora o Swagger o descreva como número da página.
> - Filtros: `nu_ano` e `id_municip` (dengue, chikungunya, zika). `id_municip` é o
>   município da **unidade notificadora** e só aceita o código IBGE de **6 dígitos**.
>   Febre amarela e UBS não têm filtros.
> - **Não use o parâmetro `campos`**: ele restringe as colunas, mas deixa a consulta muito
>   mais lenta (36 s com limit=10; HTTP 502 por timeout do gateway com limit≥100).
>   Baixe a linha inteira e selecione as colunas localmente.
> - Volume: 46.848 UBS (47 requisições, menos de 1 minuto); 2.913 casos humanos de febre
>   amarela; mais de 5 milhões de notificações de dengue só em 2024.
> - Latência: de 1 a 2 s por página de 1000 registros (cerca de 3,7 MB) com filtro de
>   município e ano; de 15 a 45 s com `offset` alto sem filtro. A carga nacional de dengue
>   pela API leva horas e transfere dezenas de GB, o que reforça usar os CSVs do
>   OpenDataSUS na carga inicial (seção 7.1).
> - Sob carga a DEMAS devolve HTTP 502 após 60 s. Mantenha a concorrência baixa (até 4)
>   e o retry com backoff.
>
> As conclusões abaixo continuam válidas: a DEMAS não serve para consultas analíticas ao
> vivo e o ETL segue necessário. O que muda é a escala do custo.

**A API DEMAS pagina com `limit` máximo de 20 registros por requisição**, usando `offset`
como número da página. Isso está documentado para `/cnes/estabelecimentos`, `/sisvan` e
`/sisagua`, e vale presumivelmente para os endpoints de arboviroses.

Implicações que moldam toda a arquitetura:

- **É inviável consultar a DEMAS ao vivo para montar gráficos.** Só as UBS do Brasil
  (~44 mil registros) exigiriam mais de 2.200 requisições sequenciais.
- O projeto **precisa** de um ETL que pagina, materializa e persiste os dados fora da API.
- Apenas consultas pontuais (1 a 5 chamadas por interação) podem ser feitas em tempo
  real pela aplicação — na prática, só o `/cnes/estabelecimentos/{codigo_cnes}`.

**A verificar antes de codar:** o catálogo de dados abertos marca vários endpoints DEMAS
como "Autenticação Necessária: Sim", com fluxo via `/autenticacao/login`. Os endpoints de
arboviroses e UBS aparentam ser abertos. Confirme com uma requisição manual antes de
assumir; se exigirem token, o ETL precisa de um passo de login e o segredo vai para as
secrets do GitHub Actions.

---

## 4. Fontes de dados

### 4.1 DEMAS — API de Dados Abertos do Ministério da Saúde

Base: `https://apidadosabertos.saude.gov.br`

| Endpoint | Uso |
|---|---|
| `/arboviroses/zikavirus` | notificações de zika |
| `/arboviroses/dengue` | notificações de dengue |
| `/arboviroses/chikungunya` | notificações de chikungunya |
| `/arboviroses/febre-amarela-humanos-primatas-nao-humanos` | febre amarela |
| `/assistencia-a-saude/unidade-basicas-de-saude` | cadastro de UBS com lat/lon |
| `/cnes/estabelecimentos/{codigo_cnes}` | detalhe de um estabelecimento |

Parâmetros comuns: `limit` (≤ 20), `offset`, e filtros como `codigo_municipio`, `uf`,
`ano_referencia`. Confirme os nomes exatos por endpoint no Swagger em
`https://apidadosabertos.saude.gov.br/v1/#/`.

Exemplo de payload de UBS:

```json
{
  "ibge": "260290",
  "bairro": "PONTE DOS CARVALHOS",
  "logradouro": "RUA DO CEMITERIO",
  "cnes": "0000302",
  "uf": "26",
  "latitude": "-8,21811",
  "longitude": "-35,22944",
  "nome": "USF SANTO ESTEVAO"
}
```

### 4.2 IBGE — Localidades

`https://servicodados.ibge.gov.br/api/v1/localidades/municipios`

Retorna os 5.570 municípios com código e nome. Usar para montar o dicionário
código ↔ nome do dropdown. **Não hardcode a lista** — materialize em Parquet no ETL.

### 4.3 Geocodificação de CEP

Ordem de preferência:

1. **BrasilAPI** — `https://brasilapi.com.br/api/cep/v2/{cep}`. MIT, gratuita, sem chave.
   Retorna `location.coordinates.latitude` e `.longitude`.
2. **AwesomeAPI** — `https://cep.awesomeapi.com.br/json/{cep}`. Retorna `lat`, `lng` e
   `city_ibge`, útil como fallback e para cruzar com o código IBGE.
3. **ViaCEP + Nominatim/OSM** via `geopy`. ViaCEP não devolve coordenadas, só endereço e
   código IBGE; o Nominatim geocodifica o endereço. Respeitar o limite de 1 req/s e
   definir um `User-Agent` identificável.

Google Maps está **descartado** por ser pago.

### 4.4 Manuais do Ministério da Saúde (corpus do RAG)

PDFs oficiais de manejo clínico e vigilância de dengue, chikungunya, zika e febre
amarela, obtidos de `bvsms.saude.gov.br` e `gov.br/saude`. Baixar e versionar as URLs
de origem, não os PDFs.

---

## 5. Arquitetura

Fluxo em quatro camadas, todo batch exceto onde indicado:

```
Fontes externas          DEMAS · IBGE · BrasilAPI · manuais PDF
        |
        v
Ingestão e processamento GitHub Actions (cron)
                         - extração e carga: httpx, Polars, DuckDB
                         - treino do modelo: scikit-learn, statsmodels
                         - indexação RAG: Docling, embeddings e5, FAISS
        |
        v
Armazenamento            Hugging Face Hub (gratuito)
                         - dados históricos: Parquet particionado
                         - modelo treinado: joblib + métricas
                         - índice vetorial: FAISS + chunks
        |
        v
Aplicação                Streamlit Community Cloud
                         - dashboard · previsão · mapa UBS · chatbot
        |
        v
Usuário
```

**Exceção em tempo real:** a consulta de CEP (BrasilAPI) e o detalhe do estabelecimento
(`/cnes/estabelecimentos/{cnes}`) são chamados direto pela aplicação, não pelo ETL. São
poucas chamadas por interação e cabem com `st.cache_data(ttl=86400)`.

O diagrama visual está em `arquitetura-arboviroses.jpg`.

---

## 6. Stack definida

### Ingestão e ETL
- `httpx` em modo async — paralelizar as milhares de páginas da DEMAS.
- `tenacity` — retry com backoff exponencial. A DEMAS instabiliza sob carga.
- `Polars` ou `DuckDB` para transformação. Ambos leem e escrevem Parquet nativamente.
- `Pandera` para validação de schema.
- Opcional: `dbt-core` + `dbt-duckdb` se quiser camada de modelagem versionada e testada.

### Orquestração
- **GitHub Actions** com `schedule` (cron). Gratuito e ilimitado em repositório público.
- Alternativas se quiser orquestrador de verdade: Prefect ou Dagster.

### Armazenamento
- **Hugging Face Hub**, repositório do tipo *dataset*, com Parquet particionado por
  doença e ano. Gratuito, versionado por Git/LFS.
- Leitura: `pd.read_parquet("hf://datasets/<user>/<repo>/...")` ou DuckDB + `httpfs`.
- O mesmo Hub guarda o modelo treinado e o índice vetorial.
- Alternativas relacionais, **não necessárias** neste projeto: Supabase (Postgres +
  PostGIS, 500 MB grátis) ou Neon.

### Aplicação
- **Streamlit** hospedado no **Streamlit Community Cloud**, ligado ao repositório GitHub:
  cada push na `main` atualiza o app.
- Por que não Hugging Face Spaces: criar Space que executa código (Gradio ou Docker) exige
  plano pago; só Spaces estáticos são gratuitos. Verificado em 2026-10-28 (HTTP 402 na
  criação do Space e nota na documentação de hardware dos Spaces). O SDK `streamlit` dos Spaces já fora descontinuado em
  2025-04-30.
- Limites do Community Cloud (documentação de fev/2024): de 690 MB a 2,7 GB de RAM, até
  2 núcleos, e o app hiberna após 12 h sem tráfego. O chatbot precisa caber nisso (ver RAG).
- Dependências do deploy em `app/requirements.txt`, gerado do `uv.lock` pelo `uv export`.
  Um arquivo na pasta do entrypoint tem precedência sobre o `uv.lock` da raiz.
- Cache: `st.cache_data` para dados, `st.cache_resource` para modelo, embedder e índice.
- Segredos em *Settings > Secrets* do app, em TOML. As chaves de nível raiz viram
  variáveis de ambiente, lidas pelo `pydantic-settings`.
- Atenção: o app **não tem disco persistente**. O índice vetorial e o modelo devem ser
  baixados do Hub no startup.

### Visualização
- **Plotly** para as séries temporais interativas.
- **Folium** + `streamlit-folium` para o mapa, com tiles OpenStreetMap (sem chave).

### Machine learning
- `scikit-learn` e `statsmodels`.
- Serialização com `joblib` ou `skops`.

### RAG
- Parsing: **Docling** (MIT, lida bem com tabelas) ou `pypdf`.
  **Evitar PyMuPDF** — é AGPL e conflita com a premissa de licenciamento.
- Embeddings: `intfloat/multilingual-e5-base` ou `BAAI/bge-m3`. Ambos MIT e fortes em
  português. O e5 exige prefixos `query: ` e `passage: `.
- Índice: **FAISS** ou **ChromaDB**, construído no CI e publicado no Hub.
- Busca híbrida: `rank_bm25` + densa, fundidas por Reciprocal Rank Fusion.
- Reranking: `BAAI/bge-reranker-v2-m3`. Com o limite de 2,7 GB do Community Cloud,
  embedder e reranker juntos não cabem com folga: preferir `multilingual-e5-small` e
  avaliar se o reranker compensa.
- Avaliação: `Ragas` (fidelidade e precisão de contexto).

### LLM
Ponto onde "gratuito + hospedado + 100% open source" não coexistem perfeitamente. Opções
honestas:
- **Groq free tier** com Llama 3.3 70B — pesos abertos, serviço proprietário, custo zero,
  latência excelente. **Escolha recomendada.**
- `llama-cpp-python` com Qwen2.5-3B quantizado rodando na CPU — totalmente autocontido,
  mas lento e com cerca de 2 GB de RAM, o que não cabe junto do embedder no Community
  Cloud. Fallback só se o Groq ficar indisponível.

### Transversal
- `uv` para dependências, `Ruff` para lint e formatação, `pytest` para testes.
- Deploy contínuo pelo Community Cloud; o CI verifica que `app/requirements.txt` acompanha o `uv.lock`.
- Custo total do projeto: R$ 0.

---

## 7. Especificação das features

### 7.1 Dashboard de dados históricos

Filtros: ano de notificação, município (dropdown por nome, resolvido para código IBGE) e
doença (dropdown que determina qual endpoint alimentou a tabela).

**Importante:** os endpoints de arboviroses retornam notificações individuais do SINAN.
Não tente ingerir o Brasil inteiro de uma vez. **Pré-agregue no ETL** por município, ano
e semana epidemiológica. Comece por um recorte — um estado, ou os N municípios maiores —
antes de expandir.

Se o volume via API ficar proibitivo, complemente a carga inicial com os CSVs do
OpenDataSUS e use a DEMAS apenas para o incremental.

### 7.2 Previsão de notificações

O pedido original era regressão linear. **Regressão linear pura sobre a contagem de
notificações vai performar mal**, por três motivos:

- são contagens: não-negativas e com sobredispersão;
- a sazonalidade anual é forte no Brasil, com pico entre fevereiro e maio;
- anos epidêmicos quebram a tendência — 2016 na zika, 2024 na dengue.

Progressão recomendada, que preserva a ideia original mas funciona:

1. **Baseline sazonal ingênuo** — mesma semana do ano anterior. Construir primeiro; é o
   número a bater.
2. **Regressão linear com engenharia de features** — termos de Fourier para sazonalidade
   (seno e cosseno da semana do ano), lags em t-1, t-4 e t-52, médias móveis. Continua
   sendo `LinearRegression`/`Ridge`, mas passa a capturar o ciclo.
3. **GLM Poisson ou binomial negativa** no `statsmodels` — verossimilhança correta para
   contagem. Ou **SARIMAX** para série temporal clássica.

Validação: `TimeSeriesSplit` ou backtest de origem móvel. **Nunca k-fold aleatório.**
Métricas: MAE e MASE — o MASE compara bem entre municípios de portes diferentes.

Na interface, rotular o resultado como **estimativa de tendência**, com intervalo de
confiança visível. Não apresentar como previsão oficial.

O modelo é treinado no CI e apenas carregado pela aplicação. **Nunca treinar em request.**

### 7.3 Mapa de UBS mais próximas

Fluxo: usuário informa o CEP → geocodificação → busca das k unidades mais próximas →
enriquecimento com `/cnes/estabelecimentos/{cnes}` → renderização no mapa.

O usuário pode escolher ver de 1 até 5 unidades mais próximas.

**Busca por proximidade:** usar `BallTree` do scikit-learn com `metric="haversine"` sobre
coordenadas em radianos. É exato, instantâneo para 44 mil pontos e dispensa PostGIS por
completo.

Renderização com Folium, popup por unidade com os dados do CNES.

Melhoria opcional: distância por rota em vez de linha reta, via OSRM público ou
OpenRouteService.

### 7.4 Chatbot RAG

Orienta o usuário a partir dos sintomas relatados, **sem diagnosticar**.

Requisitos de segurança clínica, obrigatórios:

- **Citação obrigatória** do manual e página em toda resposta.
- **Limiar de similaridade**: abaixo dele, responder "não encontrei nos manuais" em vez
  de alucinar.
- **Filtro de sinais de alarme da dengue** — dor abdominal intensa, vômitos persistentes,
  sangramento, letargia, hipotensão postural. Ao detectar, interromper o fluxo normal e
  orientar procurar atendimento imediato.
- Disclaimer visível de que a ferramenta não substitui avaliação médica.
- Registrar consultas sem resposta para melhorar o corpus.

Chunking: 500 a 800 tokens, 10 a 15% de sobreposição, cabeçalho da seção preservado nos
metadados.

---

## 8. Armadilhas conhecidas

| Armadilha | Tratamento |
|---|---|
| Coordenadas da DEMAS vêm como string com **vírgula decimal** (`"-8,21811"`) | Normalizar no ETL: `float(s.replace(",", "."))` |
| UBS com coordenada ausente ou fora do Brasil | Filtrar por bounding box: latitude entre -34 e 6, longitude entre -74 e -34 |
| `limit` da DEMAS é ≤ 20 | Paginação assíncrona com backoff; jamais consultar ao vivo para análise |
| App no Community Cloud não tem disco persistente | Baixar índice e modelo do Hub no startup |
| PyMuPDF é AGPL | Usar Docling ou pypdf |
| k-fold aleatório em série temporal | Usar `TimeSeriesSplit` |
| Códigos IBGE de 6 vs 7 dígitos | Padronizar cedo. A DEMAS usa 6 dígitos (sem dígito verificador); o IBGE usa 7 |
| `codigo_cnes` com zeros à esquerda | Tratar como string, nunca como inteiro |

---

## 9. Estrutura do repositório

```
.
├── .github/workflows/
│   ├── ci.yml               # push/PR: Ruff, pytest e sincronia do requirements.txt
│   ├── etl.yml              # cron: extração, agregação, publicação no Hub
│   ├── train.yml            # cron: retreino do modelo
│   └── index.yml            # manual: reindexação do RAG
├── notebooks/
│   └── 01_analise_exploratoria.ipynb  # etapa 0: análise qualitativa e quantitativa
├── common/
│   └── config.py            # pydantic-settings, prefixo RADAR_
├── etl/
│   ├── extract/             # clientes das APIs, paginação, retry
│   ├── transform/           # normalização, calendário epidemiológico, agregação, Pandera
│   ├── publish/             # upload para o HF Hub
│   └── pipeline.py          # ponto de entrada: python -m etl.pipeline <etapas>
├── ml/
│   ├── features.py          # Fourier, lags, médias móveis
│   ├── train.py
│   └── evaluate.py          # backtest de origem móvel, MAE/MASE
├── rag/
│   ├── fontes.yaml          # URLs dos manuais (não os PDFs)
│   ├── ingest.py            # download e parsing dos manuais
│   ├── chunk.py
│   ├── index.py             # embeddings + FAISS
│   ├── retrieve.py          # híbrido BM25 + denso, RRF, reranker
│   └── seguranca.py         # filtro de sinais de alarme
├── app/
│   ├── main.py              # st.navigation; ajusta o sys.path no Community Cloud
│   ├── requirements.txt     # dependências do deploy, geradas do uv.lock
│   ├── pages/
│   │   ├── 1_dashboard.py
│   │   ├── 2_previsao.py
│   │   ├── 3_mapa.py
│   │   └── 4_chatbot.py
│   ├── data.py              # carregamento do Hub + cache
│   └── geo.py               # CEP, BallTree, Folium
├── tests/
├── pyproject.toml           # dependências por grupo: etl, ml, app, rag, indexacao, eda, dev
└── README.md
```

---

## 10. Convenções

- Python 3.11+. Type hints em funções públicas.
- Dependências via `uv`, lint e formatação via `Ruff`.
- Configuração com `pydantic-settings`; segredos via variável de ambiente, nunca no código.
- Códigos IBGE e CNES sempre como `str`.
- Datas em ISO 8601; semana epidemiológica como campo separado, não derivada em runtime.
- Mensagens de commit e docstrings em português.

---

## 11. Ordem de implementação sugerida

0. **Análise exploratória qualitativa e quantitativa** (`notebooks/01_analise_exploratoria.ipynb`):
   conformidade da API, esquema e qualidade de cada fonte, volume e custo de extração,
   sazonalidade e sobredispersão das séries. Suas conclusões fixam as decisões de ETL
   (município de referência, semana de referência, definição de caso, recorte inicial)
   e servem de base para o modelo. O notebook não grava dados em disco.
1. Cliente da DEMAS com paginação e retry, mais teste contra um município pequeno.
2. ETL de UBS completo (dataset menor, valida o pipeline ponta a ponta) e publicação no Hub.
3. Feature 3, o mapa — é a que entrega valor mais rápido e exercita menos partes móveis.
4. ETL de arboviroses com recorte inicial, mais o dicionário de municípios do IBGE.
5. Feature 1, o dashboard.
6. Feature 2, previsão, começando pelo baseline sazonal.
7. Feature 4, o chatbot RAG.

---

## 12. Pontos em aberto

- ~~Confirmar se os endpoints de arboviroses exigem autenticação.~~ Não exigem (seção 3).
- ~~Confirmar os nomes exatos dos parâmetros de filtro por endpoint no Swagger.~~ Ver seção 3.
- Definir o município de referência da agregação: residência (`id_mn_resi`) ou
  notificação (`id_municip`). A API só filtra pela notificação.
- Definir a semana de referência (`sem_pri` ou `sem_not`) e a definição de caso
  (notificações totais ou casos prováveis), a partir do notebook de análise exploratória.
- Definir o recorte geográfico inicial da carga de arboviroses.
- Levantar a lista definitiva de manuais do MS que compõem o corpus do RAG.
- A sazonalidade varia por região: em Olinda (PE) o pico mediano da dengue foi na semana 29
  (julho), não entre fevereiro e maio (seção 7.2). Modelar a sazonalidade por região ou município.
- Investigar a ausência de notificações de zika de Olinda em 2015 e 2016 na DEMAS, apesar da
  epidemia em Pernambuco: pode haver lacuna de cobertura nos primeiros anos do agravo.
