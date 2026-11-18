"""Reranker: cross-encoder multilíngue em ONNX, sem torch.

Recebe a pergunta e os primeiros candidatos da busca híbrida e dá a cada par uma nota de
relevância, lendo os dois textos juntos (mais preciso que comparar vetores, mais caro).
Modelo: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache 2.0, treinado no mMARCO,
que inclui o português), na versão quantizada em int8.

Medido em 2026-11-17 com 37 perguntas: acerto entre os 4 primeiros de 0,81 para 0,92 e
MRR de 0,68 para 0,83, com 10 candidatos e pares de até 256 tokens (cerca de 1,5 s por
pergunta com 2 threads). Com 20 candidatos e 512 tokens, a qualidade é a mesma e o tempo
passa de 7 s.
"""

from pathlib import Path

import numpy as np


class Reranker:
    """Cross-encoder em ONNX: uma nota por par (pergunta, trecho); maior é mais relevante."""

    def __init__(
        self, modelo: Path, tokenizer: Path, max_tokens: int = 256, threads: int | None = None
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer))
        # Nos pares, a truncagem corta primeiro o texto mais longo (o trecho).
        self._tokenizer.enable_truncation(max_length=max_tokens)
        pad = self._tokenizer.token_to_id("<pad>")
        self._tokenizer.enable_padding(pad_id=pad, pad_token="<pad>")

        opcoes = ort.SessionOptions()
        opcoes.enable_cpu_mem_arena = False
        if threads:
            opcoes.intra_op_num_threads = threads
        self._sessao = ort.InferenceSession(
            str(modelo), sess_options=opcoes, providers=["CPUExecutionProvider"]
        )
        self._entradas = {e.name for e in self._sessao.get_inputs()}

    @classmethod
    def do_hub(
        cls,
        repo: str,
        arquivo: str,
        tokenizer: str,
        revisao: str | None = None,
        max_tokens: int = 256,
        threads: int | None = None,
    ) -> "Reranker":
        """Baixa (ou reaproveita do cache do Hugging Face) o modelo e o tokenizer."""
        from huggingface_hub import hf_hub_download

        modelo = hf_hub_download(repo_id=repo, filename=arquivo, revision=revisao)
        vocabulario = hf_hub_download(repo_id=repo, filename=tokenizer, revision=revisao)
        return cls(Path(modelo), Path(vocabulario), max_tokens=max_tokens, threads=threads)

    def pontuar(self, pergunta: str, textos: list[str]) -> np.ndarray:
        """Nota de relevância de cada texto para a pergunta, shape (len(textos),)."""
        if not textos:
            return np.zeros(0, dtype=np.float32)
        codificados = self._tokenizer.encode_batch([(pergunta, t) for t in textos])
        ids = np.array([c.ids for c in codificados], dtype=np.int64)
        mascara = np.array([c.attention_mask for c in codificados], dtype=np.int64)
        alimentar = {
            "input_ids": ids,
            "attention_mask": mascara,
            "token_type_ids": np.zeros_like(ids),
        }
        alimentar = {nome: valor for nome, valor in alimentar.items() if nome in self._entradas}
        return self._sessao.run(None, alimentar)[0].reshape(-1).astype(np.float32)
