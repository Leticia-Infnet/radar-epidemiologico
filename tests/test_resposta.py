import json

import httpx
import pytest

from rag.resposta import (
    MSG_LIMITE,
    MSG_NAO_ENCONTRADO,
    PROMPT_SISTEMA,
    ClienteLLM,
    LimiteDeUso,
    LLMIndisponivel,
    consulta_de_busca,
    montar_mensagens,
    responder,
    validar,
)
from rag.seguranca import ORIENTACAO_URGENTE
from tests.test_retrieve import buscador_sintetico


class LLMFalso:
    def __init__(self, texto: str = "Mantenha a hidratação [1].", erro: Exception | None = None):
        self.texto, self.erro, self.chamadas = texto, erro, []

    def completar(self, mensagens):
        self.chamadas.append(mensagens)
        if self.erro:
            raise self.erro
        return self.texto, "modelo-falso"


class BuscadorQueFalha:
    def buscar(self, *args, **kwargs):
        raise AssertionError("o alarme não deveria buscar nos manuais")


def test_validar_mantem_citacoes_validas():
    texto, citados = validar("Beba líquidos [1]. Evite AAS [2, 7]. Repouso [9].", 4)
    assert citados == [1, 2]
    assert texto == "Beba líquidos [1]. Evite AAS [2]. Repouso."


def test_validar_aceita_citacoes_coladas():
    assert validar("Hidrate-se [1][3].", 3)[1] == [1, 3]


def test_validar_descarta_sem_citacao_e_sentinela():
    assert validar("Beba bastante água.", 4) is None
    assert validar("NAO_ENCONTRADO", 4) is None
    assert validar("Texto [5].", 4) is None


def test_montar_mensagens_numera_trechos_e_corta_historico():
    historico = [
        {"papel": p, "texto": f"m{i} " + "x" * 900} for i, p in enumerate(["user", "assistant"] * 3)
    ]
    trechos = [
        {"titulo": "Manual", "edicao": "2024", "pagina_inicial": 7, "secao": "S", "texto": "T"}
    ]
    mensagens = montar_mensagens("pergunta?", historico, trechos)
    assert mensagens[0] == {"role": "system", "content": PROMPT_SISTEMA}
    assert len(mensagens) == 1 + 4 + 1  # sistema, 2 trocas, pergunta
    assert mensagens[1]["content"].startswith("m2 ")
    assert all(len(m["content"]) <= 600 for m in mensagens[1:-1])
    assert "[1] Manual (2024), página 7 do PDF — seção: S\nT" in mensagens[-1]["content"]
    assert mensagens[-1]["content"].endswith("Pergunta: pergunta?")


def test_consulta_de_busca_junta_continuacao_curta():
    historico = [{"papel": "user", "texto": "Quais os sintomas da dengue?"}]
    assert (
        consulta_de_busca("e em crianças?", historico)
        == "Quais os sintomas da dengue? e em crianças?"
    )
    longa = "Quais são os sinais de alarme da dengue em adultos idosos?"
    assert consulta_de_busca(longa, historico) == longa
    assert consulta_de_busca("e em crianças?", []) == "e em crianças?"


def test_alarme_interrompe_sem_buscar():
    resposta = responder("Minha gengiva está sangrando", [], BuscadorQueFalha(), LLMFalso())
    assert resposta.tipo == "alarme"
    assert resposta.texto == ORIENTACAO_URGENTE
    assert resposta.sinais == ["sangramento"]


def test_ignorar_alarme_segue_para_a_busca():
    llm = LLMFalso("Sangramento é sinal de alarme [1].")
    resposta = responder(
        "sangramento está entre os sinais de alarme da dengue?",
        [],
        buscador_sintetico(),
        llm,
        ignorar_alarme=True,
    )
    assert resposta.tipo == "resposta"
    assert resposta.fontes[0].numero == 1
    assert resposta.fontes[0].link == "https://exemplo/manual.pdf#page=10"
    assert resposta.modelo == "modelo-falso"


def test_abaixo_do_limiar_nao_chama_o_llm():
    llm = LLMFalso()
    resposta = responder("receita de bolo de cenoura", [], buscador_sintetico(), llm)
    assert resposta.tipo == "nao_encontrado"
    assert resposta.motivo == "abaixo_do_limiar"
    assert resposta.texto == MSG_NAO_ENCONTRADO
    assert llm.chamadas == []


@pytest.mark.parametrize(
    ("texto", "motivo"), [("NAO_ENCONTRADO", "llm_nao_encontrou"), ("Sem fonte.", "sem_citacao")]
)
def test_resposta_do_llm_sem_citacao_vira_nao_encontrado(texto, motivo):
    resposta = responder("sinais de alarme da dengue", [], buscador_sintetico(), LLMFalso(texto))
    assert (resposta.tipo, resposta.motivo) == ("nao_encontrado", motivo)


def test_limite_de_uso_mostra_trechos():
    llm = LLMFalso(erro=LimiteDeUso())
    resposta = responder("sinais de alarme da dengue", [], buscador_sintetico(), llm)
    assert (resposta.tipo, resposta.motivo, resposta.texto) == (
        "indisponivel",
        "limite",
        MSG_LIMITE,
    )
    assert 1 <= len(resposta.fontes) <= 3
    assert "Dor abdominal" in resposta.fontes[0].texto


def test_sem_llm_configurado_mostra_trechos():
    resposta = responder("sinais de alarme da dengue", [], buscador_sintetico(), None)
    assert (resposta.tipo, resposta.motivo) == ("indisponivel", "sem_llm")


def _cliente(respostas: dict[str, httpx.Response], pedidos: list) -> httpx.Client:
    def tratar(pedido: httpx.Request) -> httpx.Response:
        corpo = json.loads(pedido.content)
        pedidos.append(corpo)
        return respostas[corpo["model"]]

    return httpx.Client(transport=httpx.MockTransport(tratar))


def _ok(texto: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": texto}}]})


def test_cliente_troca_de_modelo_quando_a_cota_acaba():
    pedidos = []
    cliente = _cliente({"grande": httpx.Response(429), "pequeno": _ok("Oi [1].")}, pedidos)
    llm = ClienteLLM("https://api/v1", "chave", ["grande", "pequeno"], cliente=cliente)
    assert llm.completar([{"role": "user", "content": "x"}]) == ("Oi [1].", "pequeno")
    assert [p["model"] for p in pedidos] == ["grande", "pequeno"]
    assert pedidos[0]["reasoning_effort"] == "low"
    assert pedidos[0]["messages"] == [{"role": "user", "content": "x"}]


def test_cliente_sem_cota_em_nenhum_modelo():
    cliente = _cliente({"a": httpx.Response(429), "b": httpx.Response(429)}, [])
    with pytest.raises(LimiteDeUso):
        ClienteLLM("https://api/v1", "k", ["a", "b"], cliente=cliente).completar([])


def test_cliente_com_chave_recusada():
    cliente = _cliente({"a": httpx.Response(401), "b": _ok("x")}, [])
    with pytest.raises(LLMIndisponivel):
        ClienteLLM("https://api/v1", "k", ["a", "b"], cliente=cliente).completar([])


def test_cliente_com_resposta_vazia_tenta_o_proximo():
    cliente = _cliente({"a": _ok(""), "b": httpx.Response(500)}, [])
    with pytest.raises(LLMIndisponivel):
        ClienteLLM("https://api/v1", "k", ["a", "b"], cliente=cliente).completar([])


def test_cliente_sem_reasoning_effort():
    pedidos = []
    cliente = _cliente({"a": _ok("x [1]")}, pedidos)
    ClienteLLM("https://api/v1", "k", ["a"], reasoning_effort=None, cliente=cliente).completar([])
    assert "reasoning_effort" not in pedidos[0]
