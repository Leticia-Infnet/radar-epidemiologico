import asyncio
from types import SimpleNamespace

from rag.avaliacao_respostas import limite_por_minuto, pontuar

ERRO_MINUTO = Exception(
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute"
)
ERRO_DIA = Exception(
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-20b` on tokens per day"
)


class MetricaFalsa:
    """Falha com os erros dados, em ordem, e depois devolve 0,9."""

    def __init__(self, *erros):
        self.erros, self.chamadas = list(erros), 0

    async def ascore(self, **argumentos):
        self.chamadas += 1
        if self.erros:
            raise self.erros.pop(0)
        return SimpleNamespace(value=0.9)


def test_limite_por_minuto_e_diario():
    assert limite_por_minuto(ERRO_MINUTO)
    assert not limite_por_minuto(ERRO_DIA)
    assert not limite_por_minuto(Exception("Failed to generate JSON"))


def test_pontuar_tenta_de_novo_no_limite_por_minuto():
    metrica = MetricaFalsa(ERRO_MINUTO, ERRO_MINUTO)
    assert asyncio.run(pontuar(metrica, tentativas=3, espera=0, user_input="q")) == 0.9
    assert metrica.chamadas == 3


def test_pontuar_desiste_no_limite_diario_e_em_outros_erros():
    metrica = MetricaFalsa(ERRO_DIA)
    assert asyncio.run(pontuar(metrica, espera=0, user_input="q")) is None
    assert metrica.chamadas == 1
    metrica = MetricaFalsa(ERRO_MINUTO, ERRO_MINUTO, ERRO_MINUTO)
    assert asyncio.run(pontuar(metrica, tentativas=3, espera=0, user_input="q")) is None
    assert metrica.chamadas == 3
