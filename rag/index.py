"""Indexação dos manuais do MS para o chatbot, executada no CI (workflow `index.yml`).

Baixa os PDFs de `rag/fontes.yaml`, converte com o Docling, divide em trechos por seção,
gera os embeddings do e5 em ONNX e monta o índice FAISS. Antes de publicar, avalia a busca
com as perguntas de `rag/avaliacao.yaml`, calibra o limiar de similaridade e passa pelo
portão de qualidade. Publica em `rag/` no Hub: `indice.faiss`, `trechos.parquet` e
`metadados.json`. O app só lê esses arquivos.

Uso: `python -m rag.index [--sem-publicar] [--manuais <id> ...]`.
"""

import argparse
import json
import logging
import math
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import get_settings
from rag.chunk import dividir, filtrar_secoes, remover_repetidos, secionar, texto_indexado
from rag.ingest import Manual, baixar, carregar_fontes, converter
from rag.retrieve import COLUNAS_TRECHOS, K_CONTEXTO, Buscador
from rag.sinonimos import carregar_sinonimos
from rag.texto import normalizar

log = logging.getLogger(__name__)

AVALIACAO = Path(__file__).with_name("avaliacao.yaml")
HIT_MINIMO = 0.80  # perguntas no escopo com trecho relevante entre os K_CONTEXTO primeiros
REJEICAO_MINIMA = 0.50  # perguntas fora do escopo que ficam abaixo do limiar
TRECHOS_MINIMOS = 15  # por manual
QUANTIL_LIMIAR = 0.05  # piso das perguntas no escopo usado na calibração do limiar


class IndiceReprovado(RuntimeError):
    """O índice não passou no portão de qualidade e nada deve ser publicado."""


@dataclass
class Avaliacao:
    """Perguntas no escopo (com os manuais e termos que tornam um trecho relevante) e fora."""

    perguntas: list[dict[str, Any]]
    fora_do_escopo: list[str]


@dataclass
class Indice:
    """Tudo o que vai para a pasta `rag/` do Hub."""

    trechos: pd.DataFrame
    vetores: np.ndarray
    metadados: dict[str, Any]


def carregar_avaliacao(caminho: Path = AVALIACAO, ids: list[str] | None = None) -> Avaliacao:
    """Lê o conjunto de avaliação; com `ids`, só as perguntas que citam esses manuais."""
    import yaml

    dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    perguntas = dados["perguntas"]
    if ids is not None:
        perguntas = [
            {**p, "manuais": [m for m in p["manuais"] if m in ids]}
            for p in perguntas
            if set(p["manuais"]) & set(ids)
        ]
    return Avaliacao(perguntas, list(dados["fora_do_escopo"]))


def relevante(trecho: dict[str, Any], pergunta: dict[str, Any]) -> bool:
    """O trecho é de um dos manuais esperados e contém todos os termos da pergunta."""
    if trecho["manual"] not in pergunta["manuais"]:
        return False
    texto = normalizar(f"{trecho['secao']} {trecho['texto']}")
    return all(normalizar(termo) in texto for termo in pergunta["termos"])


def avaliar(buscador: Buscador, avaliacao: Avaliacao, k: int = K_CONTEXTO) -> dict[str, Any]:
    """Acerto entre os `k` primeiros, MRR e similaridades máximas das perguntas.

    Levanta `ValueError` se alguma pergunta não tem nenhum trecho relevante no corpus
    inteiro: aí o erro está na pergunta (termo que não aparece no manual), não na busca.
    """
    trechos = buscador.trechos.to_dict("records")
    sem_gabarito = [
        p["pergunta"] for p in avaliacao.perguntas if not any(relevante(t, p) for t in trechos)
    ]
    if sem_gabarito:
        raise ValueError(f"Perguntas sem nenhum trecho relevante no corpus: {sem_gabarito}")

    posicoes, sims_escopo, falhas = [], [], []
    for p in avaliacao.perguntas:
        busca = buscador.buscar(p["pergunta"], k=k)
        sims_escopo.append(busca.similaridade_maxima)
        acerto = next(
            (
                i
                for i, r in enumerate(busca.resultados, start=1)
                if relevante(trechos[r.posicao], p)
            ),
            None,
        )
        posicoes.append(acerto)
        if acerto is None:
            falhas.append(p["pergunta"])
    sims_fora = [buscador.buscar(q, k=1).similaridade_maxima for q in avaliacao.fora_do_escopo]
    n = len(posicoes)
    return {
        "k": k,
        "n_escopo": n,
        "n_fora": len(sims_fora),
        "hit_at_k": sum(p is not None for p in posicoes) / n if n else math.nan,
        "mrr": sum(1 / p for p in posicoes if p is not None) / n if n else math.nan,
        "sims_escopo": sims_escopo,
        "sims_fora": sims_fora,
        "falhas": falhas,
    }


def calibrar_limiar(
    sims_escopo: list[float], sims_fora: list[float], quantil: float = QUANTIL_LIMIAR
) -> tuple[float, float]:
    """Limiar de similaridade e a fração das perguntas fora do escopo abaixo dele.

    Se a maior similaridade fora do escopo fica abaixo do quantil `quantil` das perguntas
    no escopo, o limiar é o ponto médio entre os dois: perguntas coloquiais do escopo,
    um pouco menos parecidas com o texto dos manuais que as do conjunto de avaliação,
    continuam passando. Senão, é o próprio quantil.
    """
    piso_escopo = float(np.quantile(sims_escopo, quantil))
    if sims_fora and max(sims_fora) < piso_escopo:
        limiar = (max(sims_fora) + piso_escopo) / 2
    else:
        limiar = piso_escopo
    limiar = math.floor(limiar * 1000) / 1000
    rejeicao = float(np.mean([s < limiar for s in sims_fora])) if sims_fora else math.nan
    return limiar, rejeicao


def conferir_qualidade(trechos: pd.DataFrame, metricas: dict[str, Any], manuais: list[str]) -> None:
    """Portão de qualidade: levanta `IndiceReprovado` em vez de publicar um índice ruim."""
    por_manual = trechos["manual"].value_counts()
    for manual in manuais:
        if por_manual.get(manual, 0) < TRECHOS_MINIMOS:
            raise IndiceReprovado(
                f"{manual}: {por_manual.get(manual, 0)} trechos, abaixo de {TRECHOS_MINIMOS}."
            )
    if not metricas["n_escopo"] or metricas["hit_at_k"] < HIT_MINIMO:
        raise IndiceReprovado(
            f"Acerto entre os {metricas['k']} primeiros de {metricas['hit_at_k']:.2f}, "
            f"abaixo de {HIT_MINIMO}. Falhas: {metricas['falhas']}"
        )
    rejeicao = metricas["rejeicao_fora"]
    if metricas["n_fora"] and rejeicao < REJEICAO_MINIMA:
        raise IndiceReprovado(
            f"Só {rejeicao:.0%} das perguntas fora do escopo ficam abaixo do limiar "
            f"{metricas['limiar']}; o mínimo é {REJEICAO_MINIMA:.0%}."
        )


def trechos_do_manual(manual: Manual, caminho_pdf: Path, contar_tokens) -> pd.DataFrame:
    """Blocos do Docling → seções → trechos de um manual, com os dados de citação."""
    blocos = remover_repetidos(converter(caminho_pdf, manual))
    if manual.paginas and blocos and min(b.pagina for b in blocos) < manual.paginas[0]:
        raise RuntimeError(f"{manual.id}: página fora do intervalo pedido em fontes.yaml")
    trechos = dividir(filtrar_secoes(secionar(blocos)), contar_tokens)
    tabela = pd.DataFrame([t.__dict__ for t in trechos])
    return tabela.assign(titulo=manual.titulo, edicao=manual.edicao, url=manual.url)[
        COLUNAS_TRECHOS
    ]


def _validar(trechos: pd.DataFrame) -> None:
    # Importados aqui: os contratos usam Polars e Pandera, do grupo `etl`.
    import polars as pl

    from etl.transform.schemas import TrechoManual

    TrechoManual.validate(pl.from_pandas(trechos))


def gerar(ids: list[str] | None = None) -> Indice:
    """Baixa, converte, divide, indexa e avalia os manuais pedidos (todos, por padrão)."""
    import faiss
    from huggingface_hub import HfApi

    from rag.embeddings import EmbedderE5

    s = get_settings()
    manuais = carregar_fontes()
    if ids:
        desconhecidos = set(ids) - {m.id for m in manuais}
        if desconhecidos:
            raise ValueError(f"Manuais desconhecidos: {sorted(desconhecidos)}")
        manuais = [m for m in manuais if m.id in ids]

    revisao = HfApi().model_info(s.embedding_repo).sha
    embedder = EmbedderE5.do_hub(
        s.embedding_repo, s.embedding_arquivo, s.embedding_tokenizer, revisao=revisao
    )
    reranker, info_reranker = None, None
    if s.rerank_ativo:
        from rag.rerank import Reranker

        info_reranker = {
            "repo": s.rerank_repo,
            "arquivo": s.rerank_arquivo,
            "tokenizer": s.rerank_tokenizer,
            "revisao": HfApi().model_info(s.rerank_repo).sha,
            "candidatos": s.rerank_candidatos,
            "max_tokens": s.rerank_max_tokens,
        }
        reranker = Reranker.do_hub(
            s.rerank_repo,
            s.rerank_arquivo,
            s.rerank_tokenizer,
            revisao=info_reranker["revisao"],
            max_tokens=s.rerank_max_tokens,
        )

    partes, info_manuais = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for manual in manuais:
            caminho, sha256 = baixar(manual, Path(tmp))
            trechos = trechos_do_manual(manual, caminho, embedder.contar_tokens)
            caminho.unlink()
            log.info("%s: %d trechos.", manual.id, len(trechos))
            partes.append(trechos)
            info_manuais[manual.id] = {
                "titulo": manual.titulo,
                "edicao": manual.edicao,
                "doencas": list(manual.doencas),
                "url": manual.url,
                "paginas": list(manual.paginas) if manual.paginas else None,
                "sha256": sha256,
                "trechos": len(trechos),
            }
    trechos = pd.concat(partes, ignore_index=True)
    _validar(trechos)

    textos = [texto_indexado(t.titulo, t.secao, t.texto) for t in trechos.itertuples()]
    vetores = embedder.passagens(textos)
    indice = faiss.IndexFlatIP(vetores.shape[1])
    indice.add(vetores)

    # O portão avalia a busca como o app a fará, com o reranker e os sinônimos.
    sinonimos = carregar_sinonimos()
    buscador = Buscador(
        trechos,
        indice,
        embedder,
        limiar=0.0,
        reranker=reranker,
        candidatos_rerank=s.rerank_candidatos,
        sinonimos=sinonimos,
    )
    metricas = avaliar(buscador, carregar_avaliacao(ids=ids))
    limiar, rejeicao = calibrar_limiar(metricas["sims_escopo"], metricas["sims_fora"])
    metricas.update(limiar=limiar, rejeicao_fora=rejeicao)
    log.info(
        "Avaliação: acerto@%d %.2f, MRR %.2f, limiar %.3f, rejeição fora do escopo %.0f%%, "
        "falhas %s",
        metricas["k"],
        metricas["hit_at_k"],
        metricas["mrr"],
        limiar,
        100 * rejeicao,
        metricas["falhas"],
    )
    conferir_qualidade(trechos, metricas, [m.id for m in manuais])

    metadados = {
        "tabela": "rag",
        "linhas": len(trechos),
        "gerado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding": {
            "repo": s.embedding_repo,
            "arquivo": s.embedding_arquivo,
            "tokenizer": s.embedding_tokenizer,
            "revisao": revisao,
            "dimensao": int(vetores.shape[1]),
        },
        "reranker": info_reranker,
        "sinonimos": len(sinonimos),
        "chunking": {"max_tokens": int(trechos["n_tokens"].max())},
        "limiar_similaridade": limiar,
        "avaliacao": {
            "k": metricas["k"],
            "hit_at_k": round(metricas["hit_at_k"], 4),
            "mrr": round(metricas["mrr"], 4),
            "n_escopo": metricas["n_escopo"],
            "n_fora": metricas["n_fora"],
            "rejeicao_fora": round(rejeicao, 4),
            "falhas": metricas["falhas"],
        },
        "manuais": info_manuais,
        "versoes": {p: version(p) for p in ("docling", "faiss-cpu", "onnxruntime", "tokenizers")},
    }
    return Indice(trechos=trechos, vetores=vetores, metadados=metadados)


def executar(publicar: bool = True, ids: list[str] | None = None) -> Indice:
    """Gera o índice e o publica em `rag/` num commit só."""
    resultado = gerar(ids)
    log.info("rag: %s", json.dumps(resultado.metadados, ensure_ascii=False))
    if not publicar:
        log.info("rag: validado, publicação desativada (--sem-publicar).")
        return resultado
    if ids:
        raise SystemExit("Publicar só com o corpus inteiro: rode sem --manuais.")

    import faiss

    from etl.publish.hub import publicar_pasta

    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        indice = faiss.IndexFlatIP(resultado.vetores.shape[1])
        indice.add(resultado.vetores)
        faiss.write_index(indice, str(pasta / "indice.faiss"))
        resultado.trechos.to_parquet(pasta / "trechos.parquet", index=False)
        (pasta / "metadados.json").write_text(
            json.dumps(resultado.metadados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        commit = publicar_pasta(
            pasta,
            "rag",
            f"rag: {len(resultado.trechos)} trechos",
            apagar=["*.faiss", "*.parquet"],
        )
    log.info("rag: publicado em %s (commit %s).", get_settings().hf_dataset_repo, commit)
    return resultado


def main() -> None:
    """Indexa os manuais, avalia a busca e publica o índice no Hub."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-publicar", action="store_true", help="Indexa e valida sem enviar.")
    parser.add_argument("--manuais", nargs="+", help="Só estes ids (teste rápido; não publica).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    executar(publicar=not args.sem_publicar, ids=args.manuais)


if __name__ == "__main__":
    main()
