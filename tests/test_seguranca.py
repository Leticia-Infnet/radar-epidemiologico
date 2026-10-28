import pytest

from rag.seguranca import sinais_detectados


@pytest.mark.parametrize(
    ("mensagem", "sinal"),
    [
        ("Estou com dor abdominal intensa desde ontem", "dor abdominal intensa"),
        ("meu filho não para de vomitar", "vômitos persistentes"),
        ("minha gengiva está sangrando", "sangramento"),
        ("Ela está muito sonolenta e não consegue ficar acordada", "letargia"),
        ("sinto tontura ao levantar", "hipotensão postural"),
    ],
)
def test_detecta_sinais_de_alarme(mensagem, sinal):
    assert sinal in sinais_detectados(mensagem)


def test_mensagem_sem_sinais():
    assert sinais_detectados("Tenho febre e dor no corpo há dois dias") == []
