import numpy as np
import pytest

from rag.embeddings import pooling_medio


def test_pooling_ignora_posicoes_mascaradas_e_normaliza():
    estado = np.array([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]])
    mascara = np.array([[1, 1, 0]])
    vetor = pooling_medio(estado, mascara)
    np.testing.assert_allclose(vetor, [[1.0, 0.0]])
    assert vetor.dtype == np.float32


@pytest.mark.integracao
def test_e5_real_aproxima_pergunta_do_trecho_certo():
    from common.config import get_settings
    from rag.embeddings import EmbedderE5

    s = get_settings()
    embedder = EmbedderE5.do_hub(s.embedding_repo, s.embedding_arquivo, s.embedding_tokenizer)
    trechos = embedder.passagens(
        [
            "Os sinais de alarme da dengue incluem dor abdominal intensa e vômitos persistentes.",
            "A vacina da febre amarela é aplicada em dose única.",
        ]
    )
    similaridades = trechos @ embedder.consulta("quais os sinais de alarme da dengue")
    assert similaridades[0] > similaridades[1]
    assert embedder.contar_tokens("dengue") > 0
