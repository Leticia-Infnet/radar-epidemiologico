import hashlib

import faiss
import numpy as np
import pandas as pd
import pytest

from rag.chunk import texto_indexado
from rag.retrieve import Buscador, rrf
from rag.texto import normalizar, tokenizar

DIMENSAO = 512


class EmbedderFalso:
    """Saco de palavras com hash: textos com as mesmas palavras ficam próximos."""

    def _vetor(self, texto: str) -> np.ndarray:
        v = np.zeros(DIMENSAO, dtype=np.float32)
        for palavra in tokenizar(texto):
            v[int(hashlib.md5(palavra.encode()).hexdigest(), 16) % DIMENSAO] += 1.0
        norma = np.linalg.norm(v)
        return v / norma if norma else v

    def consulta(self, texto: str) -> np.ndarray:
        return self._vetor(texto)

    def passagens(self, textos: list[str]) -> np.ndarray:
        return np.vstack([self._vetor(t) for t in textos])


def trechos_sinteticos() -> pd.DataFrame:
    linhas = [
        ("dengue", "Sinais de alarme", "Dor abdominal intensa, vômitos persistentes e letargia."),
        ("dengue", "Hidratação", "Hidratação oral com soro e líquidos caseiros em casa."),
        ("febre", "Vacinação", "A vacina contra febre amarela é aplicada em dose única."),
        ("chik", "Fase crônica", "Artralgia persistente após três meses; usar código CID M65."),
    ]
    return pd.DataFrame(
        {
            "id": [f"{m}-{i:04d}" for i, (m, _, _) in enumerate(linhas)],
            "manual": [m for m, _, _ in linhas],
            "titulo": ["Manual " + m for m, _, _ in linhas],
            "edicao": "2024",
            "url": "https://exemplo/manual.pdf",
            "pagina_inicial": [10, 20, 30, 40],
            "pagina_final": [10, 21, 30, 41],
            "secao": [s for _, s, _ in linhas],
            "texto": [t for _, _, t in linhas],
            "n_tokens": 50,
        }
    )


def buscador_sintetico(limiar: float = 0.3) -> Buscador:
    """Buscador pequeno, com índice FAISS de verdade sobre o embedder falso."""
    trechos = trechos_sinteticos()
    embedder = EmbedderFalso()
    vetores = embedder.passagens(
        [texto_indexado(t.titulo, t.secao, t.texto) for t in trechos.itertuples()]
    )
    indice = faiss.IndexFlatIP(DIMENSAO)
    indice.add(vetores)
    return Buscador(trechos, indice, embedder, limiar=limiar)


def test_tokenizar_tira_acentos_e_stopwords():
    assert tokenizar("Quais são os sinais de alarme da Dengue?") == ["sinais", "alarme", "dengue"]
    assert normalizar("  Vômitos\n PERSISTENTES ") == "vomitos persistentes"


def test_rrf_soma_inversos_das_posicoes():
    pontos = rrf([[1, 2], [2, 3]], k=60)
    assert pontos[2] == pytest.approx(1 / 62 + 1 / 61)
    assert pontos[1] == pytest.approx(1 / 61)
    assert max(pontos, key=pontos.get) == 2


def test_busca_traz_o_trecho_certo_em_primeiro():
    busca = buscador_sintetico().buscar("quais os sinais de alarme da dengue?", k=2)
    assert busca.encontrou
    assert busca.resultados[0].posicao == 0
    assert busca.resultados[0].similaridade == pytest.approx(busca.similaridade_maxima)


def test_termo_raro_entra_pelo_bm25():
    buscador = buscador_sintetico()
    busca = buscador.buscar("M65", k=4)
    assert 3 in [r.posicao for r in busca.resultados]


def test_pergunta_fora_do_tema_fica_abaixo_do_limiar():
    busca = buscador_sintetico().buscar("receita de bolo de cenoura com chocolate")
    assert not busca.encontrou


def test_indice_e_trechos_precisam_coincidir():
    trechos = trechos_sinteticos()
    with pytest.raises(ValueError):
        Buscador(trechos, faiss.IndexFlatIP(DIMENSAO), EmbedderFalso(), limiar=0.5)


class RerankerFalso:
    """Dá a maior nota aos textos que contêm `preferido`; guarda o que recebeu."""

    def __init__(self, preferido: str):
        self.preferido, self.chamadas = preferido, []

    def pontuar(self, pergunta, textos):
        self.chamadas.append(list(textos))
        return np.array([float(self.preferido in t) for t in textos], dtype=np.float32)


def test_reranker_reordena_os_candidatos():
    base = buscador_sintetico()
    reranker = RerankerFalso("Hidratação")
    buscador = Buscador(
        base.trechos,
        base.indice,
        base.embedder,
        base.limiar,
        reranker=reranker,
        candidatos_rerank=3,
    )
    sem = base.buscar("sinais de alarme da dengue e hidratação", k=2)
    com = buscador.buscar("sinais de alarme da dengue e hidratação", k=2)
    assert sem.resultados[0].posicao == 0  # sem reranker, os sinais de alarme vêm primeiro
    assert com.resultados[0].posicao == 1  # o reranker sobe a hidratação
    assert com.resultados[0].pontuacao == 1.0
    assert len(reranker.chamadas[0]) <= 3  # só os primeiros candidatos vão ao reranker
    # A similaridade densa e o limiar não dependem do reranker.
    assert com.similaridade_maxima == sem.similaridade_maxima
    assert com.encontrou == sem.encontrou


@pytest.mark.integracao
def test_reranker_real_prefere_o_trecho_certo():
    from common.config import get_settings
    from rag.rerank import Reranker

    s = get_settings()
    reranker = Reranker.do_hub(
        s.rerank_repo, s.rerank_arquivo, s.rerank_tokenizer, max_tokens=s.rerank_max_tokens
    )
    notas = reranker.pontuar(
        "Quais remédios para dor e febre podem ser usados na dengue?",
        [
            "A febre na dengue dura de dois a sete dias e costuma ser alta.",
            "Para dor e febre, usar dipirona ou paracetamol; evitar anti-inflamatórios.",
        ],
    )
    assert notas[1] > notas[0]
