"""Chatbot contra o Hub e o LLM reais. Rodam só com `uv run pytest -m integracao`."""

import pytest

from common.config import get_settings
from rag.resposta import ClienteLLM, responder
from rag.retrieve import carregar_buscador

pytestmark = pytest.mark.integracao


@pytest.fixture(scope="module")
def buscador():
    return carregar_buscador()


def test_indice_publicado_acha_os_sinais_de_alarme(buscador):
    busca = buscador.buscar("Quais são os sinais de alarme da dengue?")
    assert busca.encontrou
    primeiro = buscador.trecho(busca.resultados[0].posicao)
    assert primeiro["manual"] in {"dengue-manejo-clinico", "guia-vigilancia-arboviroses"}


def test_indice_publicado_recusa_pergunta_fora_do_tema(buscador):
    assert not buscador.buscar("Qual é a receita de bolo de cenoura?").encontrou


@pytest.mark.skipif(get_settings().llm_api_key is None, reason="sem RADAR_LLM_API_KEY")
def test_llm_real_responde_com_citacao(buscador):
    resposta = responder(
        "Posso tomar anti-inflamatório com suspeita de dengue?",
        [],
        buscador,
        ClienteLLM.da_configuracao(),
    )
    assert resposta.tipo == "resposta", resposta
    assert resposta.fontes
