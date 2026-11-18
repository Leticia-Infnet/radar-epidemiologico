from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import data
from rag.resposta import MSG_LIMITE, LimiteDeUso
from rag.seguranca import ORIENTACAO_URGENTE
from tests.test_retrieve import buscador_sintetico

PAGINA = str(Path(__file__).parents[1] / "app" / "pages" / "4_chatbot.py")


class LLMFalso:
    def __init__(self, texto="Beba bastante líquido [1].", erro=None):
        self.texto, self.erro, self.chamadas = texto, erro, 0

    def completar(self, mensagens):
        self.chamadas += 1
        if self.erro:
            raise self.erro
        return self.texto, "modelo-falso"


class RegistroFalso:
    def __init__(self):
        self.chamadas = []

    def registrar(self, pergunta, motivo, similaridade):
        self.chamadas.append((pergunta, motivo))


@pytest.fixture
def chatbot(monkeypatch):
    """Buscador sintético, LLM e registro falsos; devolve os falsos para conferência."""
    st.cache_data.clear()
    buscador, llm, registro = buscador_sintetico(), LLMFalso(), RegistroFalso()
    monkeypatch.setattr(data, "buscador_rag", lambda: buscador)
    monkeypatch.setattr(data, "cliente_llm", lambda: llm)
    monkeypatch.setattr(data, "registro_consultas", lambda: registro)
    return llm, registro


def _perguntar(texto: str, app: AppTest | None = None) -> AppTest:
    app = app or AppTest.from_file(PAGINA, default_timeout=30).run()
    app.chat_input[0].set_value(texto)
    return app.run()


def _textos(app: AppTest) -> str:
    return " ".join(m.value for m in app.markdown)


def test_abre_com_aviso_e_exemplos(chatbot):
    app = AppTest.from_file(PAGINA, default_timeout=30).run()
    assert not app.exception
    assert "não substitui avaliação médica" in app.warning[0].value
    assert any("sinais de alarme" in b.label for b in app.button)


def test_resposta_com_fontes_e_link(chatbot):
    app = _perguntar("Quais são os sinais de alarme da dengue?")
    assert not app.exception
    textos = _textos(app)
    assert "Beba bastante líquido [1]." in textos
    assert "**Fontes**" in textos
    assert "https://exemplo/manual.pdf#page=10" in textos
    assert "p. 10 do PDF" in textos


def test_exemplo_responde_ao_clicar(chatbot):
    llm, _ = chatbot
    app = AppTest.from_file(PAGINA, default_timeout=30).run()
    app.button(key="exemplo_Quais são os sinais de alarme da dengue?").click()
    app.run()
    assert not app.exception
    assert llm.chamadas == 1
    assert "Fontes" in _textos(app)


def test_alarme_interrompe_e_botao_segue(chatbot):
    llm, _ = chatbot
    app = _perguntar("Sangramento está entre os sinais de alarme da dengue?")
    assert ORIENTACAO_URGENTE in app.error[0].value
    assert "sangramento" in app.error[0].value
    assert llm.chamadas == 0
    app.button(key="seguir_1").click()
    app.run()
    assert not app.exception
    assert llm.chamadas == 1
    assert "Beba bastante líquido [1]." in _textos(app)
    assert not [b for b in app.button if b.key == "seguir_1"]


def test_fora_do_tema_registra_a_pergunta(chatbot):
    llm, registro = chatbot
    app = _perguntar("receita de bolo de cenoura com chocolate")
    assert "Não encontrei" in app.info[0].value
    assert registro.chamadas == [("receita de bolo de cenoura com chocolate", "abaixo_do_limiar")]
    assert llm.chamadas == 0


def test_cota_esgotada_mostra_trechos(chatbot, monkeypatch):
    monkeypatch.setattr(data, "cliente_llm", lambda: LLMFalso(erro=LimiteDeUso()))
    app = _perguntar("sinais de alarme da dengue")
    assert not app.exception
    assert MSG_LIMITE in [w.value for w in app.warning]
    assert "> Dor abdominal intensa" in _textos(app)


def test_sem_indice_avisa(monkeypatch):
    monkeypatch.setattr(data, "buscador_rag", lambda: None)
    app = AppTest.from_file(PAGINA, default_timeout=30).run()
    assert not app.exception
    assert "indisponível" in app.info[0].value
