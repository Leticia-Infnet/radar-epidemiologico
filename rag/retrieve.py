"""Busca híbrida: BM25 + densa (e5, prefixo `query: `), fundidas por Reciprocal Rank Fusion,
com reranking dos primeiros candidatos por um cross-encoder (`rag.rerank`), se houver.

O limiar vale para a maior similaridade densa (cosseno) da pergunta com algum trecho, não
para a nota do reranker: abaixo dele, o chatbot responde que não encontrou nos manuais.
Ele é calibrado no CI (`rag.index`) e publicado no `metadados.json`.

O `bge-reranker-v2-m3` não cabe na RAM do Streamlit Community Cloud junto com o embedder;
o reranker usado é um MiniLM multilíngue bem menor.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from rag.chunk import texto_indexado
from rag.sinonimos import Sinonimo, expandir
from rag.texto import tokenizar

K_CONTEXTO = 4  # trechos enviados ao LLM
CANDIDATOS = 20  # de cada busca, antes da fusão
RRF_K = 60

COLUNAS_TRECHOS = [
    "id",
    "manual",
    "titulo",
    "edicao",
    "url",
    "pagina_inicial",
    "pagina_final",
    "secao",
    "texto",
    "n_tokens",
]


class Embedder(Protocol):
    def consulta(self, texto: str) -> np.ndarray: ...


class Reordenador(Protocol):
    def pontuar(self, pergunta: str, textos: list[str]) -> np.ndarray: ...


class IndiceVetorial(Protocol):
    """O que se usa do `faiss.Index`."""

    ntotal: int

    def search(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class Resultado:
    posicao: int  # linha em `trechos`
    similaridade: float  # cosseno denso; 0 se o trecho só veio do BM25
    pontuacao: float  # nota do reranker, ou RRF sem reranker


@dataclass(frozen=True)
class Busca:
    resultados: list[Resultado]
    similaridade_maxima: float
    encontrou: bool  # similaridade_maxima >= limiar


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: soma de 1 / (k + posição) em cada ranking (posição desde 1)."""
    pontos: dict[int, float] = {}
    for ranking in rankings:
        for posicao, item in enumerate(ranking, start=1):
            pontos[item] = pontos.get(item, 0.0) + 1.0 / (k + posicao)
    return pontos


class Buscador:
    """Busca nos trechos dos manuais; compartilhado entre as sessões do app."""

    def __init__(
        self,
        trechos: pd.DataFrame,
        indice: IndiceVetorial,
        embedder: Embedder,
        limiar: float,
        reranker: Reordenador | None = None,
        candidatos_rerank: int = 10,
        sinonimos: list[Sinonimo] | None = None,
    ) -> None:
        from rank_bm25 import BM25Okapi

        if indice.ntotal != len(trechos):
            raise ValueError(f"índice com {indice.ntotal} vetores para {len(trechos)} trechos")
        self.trechos = trechos.reset_index(drop=True)
        self.indice = indice
        self.embedder = embedder
        self.limiar = limiar
        self.reranker = reranker
        self.candidatos_rerank = candidatos_rerank
        self.sinonimos = sinonimos or []
        self._textos = [
            texto_indexado(t.titulo, t.secao, t.texto) for t in self.trechos.itertuples()
        ]
        self._bm25 = BM25Okapi([tokenizar(texto) for texto in self._textos])

    def trecho(self, posicao: int) -> dict[str, Any]:
        """Linha de `trechos` como dicionário."""
        return self.trechos.iloc[posicao].to_dict()

    def buscar(self, pergunta: str, k: int = K_CONTEXTO, candidatos: int = CANDIDATOS) -> Busca:
        """Os `k` trechos mais relevantes e se a pergunta passa do limiar.

        As duas buscas usam a pergunta expandida com os sinônimos técnicos; o limiar e o
        reranker, a original (o limiar foi calibrado sobre ela).
        """
        candidatos = min(candidatos, len(self.trechos))
        consulta = expandir(pergunta, self.sinonimos)
        vetor = np.asarray(self.embedder.consulta(consulta), dtype=np.float32)
        sims, ids = self.indice.search(vetor[None, :], candidatos)
        denso = [int(i) for i in ids[0] if i >= 0]
        similaridade = {int(i): float(s) for i, s in zip(ids[0], sims[0], strict=True) if i >= 0}
        if consulta == pergunta:
            similaridade_maxima = max(similaridade.values(), default=0.0)
        else:
            original = np.asarray(self.embedder.consulta(pergunta), dtype=np.float32)
            similaridade_maxima = float(self.indice.search(original[None, :], 1)[0][0][0])

        notas = self._bm25.get_scores(tokenizar(consulta))
        ordem = np.argsort(-notas, kind="stable")[:candidatos]
        lexico = [int(i) for i in ordem if notas[i] > 0]

        fundidos = sorted(rrf([denso, lexico]).items(), key=lambda par: -par[1])
        if self.reranker is not None:
            posicoes = [posicao for posicao, _ in fundidos[: self.candidatos_rerank]]
            notas_rerank = self.reranker.pontuar(pergunta, [self._textos[i] for i in posicoes])
            ordem = np.argsort(-np.asarray(notas_rerank), kind="stable")
            fundidos = [(posicoes[i], float(notas_rerank[i])) for i in ordem]
        resultados = [
            Resultado(posicao, similaridade.get(posicao, 0.0), pontos)
            for posicao, pontos in fundidos[:k]
        ]
        return Busca(resultados, similaridade_maxima, similaridade_maxima >= self.limiar)


def carregar_buscador(settings: Settings | None = None, threads: int | None = 2) -> Buscador:
    """Buscador com o índice, os trechos e o embedder publicados por `rag.index` no Hub.

    O embedder e o reranker são os da revisão registrada no `metadados.json`, os mesmos
    avaliados no portão de qualidade do índice. `RADAR_RERANK_ATIVO=false` desliga o
    reranker no app (por memória, por exemplo). Levanta `FileNotFoundError`, `OSError` ou
    `KeyError` se algo faltar no Hub.
    """
    import json

    import faiss
    from huggingface_hub import HfFileSystem, hf_hub_download

    from rag.embeddings import EmbedderE5

    s = settings or get_settings()
    base = f"datasets/{s.hf_dataset_repo}/rag"
    info = json.loads(HfFileSystem().read_text(f"{base}/metadados.json", encoding="utf-8"))
    trechos = pd.read_parquet(f"hf://{base}/trechos.parquet")
    caminho = hf_hub_download(
        repo_id=s.hf_dataset_repo, repo_type="dataset", filename="rag/indice.faiss"
    )
    indice = faiss.read_index(caminho)
    embedding = info["embedding"]
    embedder = EmbedderE5.do_hub(
        embedding["repo"],
        embedding["arquivo"],
        embedding["tokenizer"],
        revisao=embedding["revisao"],
        threads=threads,
    )
    reranker, candidatos = None, 10
    if (info_rerank := info.get("reranker")) and s.rerank_ativo:
        from rag.rerank import Reranker

        reranker = Reranker.do_hub(
            info_rerank["repo"],
            info_rerank["arquivo"],
            info_rerank["tokenizer"],
            revisao=info_rerank["revisao"],
            max_tokens=info_rerank["max_tokens"],
            threads=threads,
        )
        candidatos = info_rerank["candidatos"]
    from rag.sinonimos import carregar_sinonimos

    # Índice pequeno: uma thread evita disputa de OpenMP com o onnxruntime.
    faiss.omp_set_num_threads(1)
    return Buscador(
        trechos,
        indice,
        embedder,
        limiar=info["limiar_similaridade"],
        reranker=reranker,
        candidatos_rerank=candidatos,
        sinonimos=carregar_sinonimos(),
    )
