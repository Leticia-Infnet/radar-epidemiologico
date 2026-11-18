"""Embeddings do multilingual-e5 em ONNX, sem torch.

O mesmo código gera os vetores dos trechos no CI e o da pergunta no app, então os dois
lados são idênticos. Reproduz o sentence-transformers: mean pooling sobre a máscara de
atenção e normalização L2. O e5 exige os prefixos `passage: ` e `query: `.
"""

from pathlib import Path

import numpy as np


def pooling_medio(ultimo_estado: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Média dos vetores dos tokens reais, normalizada em L2: (n, seq, dim) → (n, dim)."""
    pesos = mascara[..., None].astype(np.float32)
    soma = (ultimo_estado.astype(np.float32) * pesos).sum(axis=1)
    medias = soma / np.clip(pesos.sum(axis=1), 1e-9, None)
    normas = np.linalg.norm(medias, axis=1, keepdims=True)
    return (medias / np.clip(normas, 1e-12, None)).astype(np.float32)


class EmbedderE5:
    """multilingual-e5 em ONNX com o tokenizer do próprio modelo."""

    def __init__(
        self, modelo: Path, tokenizer: Path, max_tokens: int = 512, threads: int | None = None
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # Um tokenizer só: o do e5 ocupa cerca de 300 MB de RAM.
        self._tokenizer = Tokenizer.from_file(str(tokenizer))
        self._tokenizer.enable_truncation(max_length=max_tokens)
        pad = self._tokenizer.token_to_id("<pad>")
        self._tokenizer.enable_padding(pad_id=pad, pad_token="<pad>")

        opcoes = ort.SessionOptions()
        # Sem a arena de memória da CPU, que guarda o pico de cada lote: no app, só perguntas
        # curtas; no CI, lotes grandes. Medido em 2026-11-17: cerca de 150 MB a menos.
        opcoes.enable_cpu_mem_arena = False
        if threads:
            opcoes.intra_op_num_threads = threads
        self._sessao = ort.InferenceSession(
            str(modelo), sess_options=opcoes, providers=["CPUExecutionProvider"]
        )
        self._entradas = {e.name for e in self._sessao.get_inputs()}
        self.dimensao = int(self._sessao.get_outputs()[0].shape[-1])

    @classmethod
    def do_hub(
        cls,
        repo: str,
        arquivo: str,
        tokenizer: str,
        revisao: str | None = None,
        threads: int | None = None,
    ) -> "EmbedderE5":
        """Baixa (ou reaproveita do cache do Hugging Face) o modelo e o tokenizer."""
        from huggingface_hub import hf_hub_download

        modelo = hf_hub_download(repo_id=repo, filename=arquivo, revision=revisao)
        vocabulario = hf_hub_download(repo_id=repo, filename=tokenizer, revision=revisao)
        return cls(Path(modelo), Path(vocabulario), threads=threads)

    def contar_tokens(self, texto: str) -> int:
        """Tokens do texto, sem os especiais.

        Limitado a `max_tokens` pela truncagem do tokenizer: basta para o chunking, que só
        precisa saber se um pedaço passa de `chunk.MAX_TOKENS`.
        """
        return len(self._tokenizer.encode(texto, add_special_tokens=False).ids)

    def _codificar(self, textos: list[str]) -> np.ndarray:
        codificados = self._tokenizer.encode_batch(textos)
        ids = np.array([c.ids for c in codificados], dtype=np.int64)
        mascara = np.array([c.attention_mask for c in codificados], dtype=np.int64)
        alimentar = {"input_ids": ids, "attention_mask": mascara}
        if "token_type_ids" in self._entradas:
            alimentar["token_type_ids"] = np.zeros_like(ids)
        alimentar = {nome: valor for nome, valor in alimentar.items() if nome in self._entradas}
        ultimo_estado = self._sessao.run(None, alimentar)[0]
        return pooling_medio(ultimo_estado, mascara)

    def passagens(self, textos: list[str], lote: int = 16) -> np.ndarray:
        """Vetores dos trechos do corpus, com o prefixo `passage: `."""
        if not textos:
            return np.zeros((0, self.dimensao), dtype=np.float32)
        partes = [
            self._codificar([f"passage: {t}" for t in textos[i : i + lote]])
            for i in range(0, len(textos), lote)
        ]
        return np.vstack(partes)

    def consulta(self, texto: str) -> np.ndarray:
        """Vetor da pergunta, com o prefixo `query: `; shape (dimensao,)."""
        return self._codificar([f"query: {texto}"])[0]
