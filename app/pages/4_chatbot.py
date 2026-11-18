"""Chatbot de orientação sobre os manuais do Ministério da Saúde (AGENTS.md, seção 7.4).

A lógica fica em `rag.resposta` (filtro de alarme, busca, LLM e citações); esta página só
cuida da conversa e da exibição. O índice é construído no CI (`rag.index`) e lido do Hub.
"""

import re

import streamlit as st

from app import data, estilo
from rag.resposta import MSG_NAO_ENCONTRADO, Fonte, Resposta, responder
from rag.texto import normalizar

MAX_PERGUNTAS = 20  # por sessão: protege a cota gratuita do LLM
MAX_CARACTERES_TRECHO = 600
EXEMPLOS = (
    "Quais são os sinais de alarme da dengue?",
    "Como diferenciar dengue, chikungunya e zika?",
    "Zika na gravidez traz riscos?",
    "Quais são os sintomas da febre amarela?",
)
BOTAO_SEGUIR = "Não estou com esse sintoma, quero só a informação"


class _NaoCachear(Exception):
    """Resposta que não deve ficar no cache (LLM indisponível ou sem cota)."""

    def __init__(self, resposta: Resposta) -> None:
        self.resposta = resposta


def _sem_markdown(texto: str) -> str:
    """Escapa caracteres que o Markdown interpretaria em textos vindos dos PDFs."""
    return re.sub(r"([\\`*_{}\[\]<>#|~])", r"\\\1", texto)


@st.cache_data(ttl=86400, max_entries=200, show_spinner=False)
def _primeira_resposta(
    chave: str, ignorar_alarme: bool, _pergunta: str, _buscador, _llm
) -> Resposta:
    """A primeira pergunta de uma conversa não depende do histórico: as repetidas (como as
    dos exemplos) não gastam a cota do LLM de novo."""
    resposta = responder(_pergunta, [], _buscador, _llm, ignorar_alarme)
    if resposta.tipo == "indisponivel":
        raise _NaoCachear(resposta)
    return resposta


def _historico(conversa: list[dict]) -> list[dict[str, str]]:
    """Trocas anteriores que tiveram resposta dos manuais, no formato de `responder`."""
    historico = []
    for m in conversa:
        if m["papel"] == "assistant" and m["resposta"].tipo == "resposta":
            historico.append({"papel": "user", "texto": m["pergunta"]})
            historico.append({"papel": "assistant", "texto": m["resposta"].texto})
    return historico


def _responder(pergunta: str, conversa: list[dict], ignorar_alarme: bool) -> Resposta:
    buscador, llm = data.buscador_rag(), data.cliente_llm()
    historico = _historico(conversa)
    if historico:
        return responder(pergunta, historico, buscador, llm, ignorar_alarme)
    try:
        return _primeira_resposta(normalizar(pergunta), ignorar_alarme, pergunta, buscador, llm)
    except _NaoCachear as exc:
        return exc.resposta


def _processar(pergunta: str, ignorar_alarme: bool = False) -> None:
    """Responde, guarda na conversa e registra as perguntas sem resposta."""
    conversa = st.session_state["conversa"]
    anteriores = list(conversa)
    if not ignorar_alarme:
        conversa.append({"papel": "user", "texto": pergunta})
    with st.chat_message("assistant"), st.spinner("Consultando os manuais..."):
        resposta = _responder(pergunta, anteriores, ignorar_alarme)
    conversa.append({"papel": "assistant", "pergunta": pergunta, "resposta": resposta})
    if resposta.tipo == "nao_encontrado":
        data.registro_consultas().registrar(pergunta, resposta.motivo, resposta.similaridade)
    st.rerun()


def _paginas(fonte: Fonte) -> str:
    if fonte.pagina_inicial == fonte.pagina_final:
        return f"p. {fonte.pagina_inicial}"
    return f"p. {fonte.pagina_inicial}–{fonte.pagina_final}"


def _citacao(fonte: Fonte) -> str:
    return (
        f"[{fonte.numero}] **{fonte.titulo}** ({fonte.edicao}), {_paginas(fonte)} do PDF — "
        f"seção: {_sem_markdown(fonte.secao)} — [abrir]({fonte.link})"
    )


def _mostrar_resposta(indice: int, mensagem: dict, ultima: bool) -> bool:
    """Mostra a resposta; devolve True se o usuário pediu para seguir depois de um alerta."""
    resposta: Resposta = mensagem["resposta"]
    if resposta.tipo == "alarme":
        sinais = ", ".join(resposta.sinais)
        st.error(f"{resposta.texto}\n\n**Sinais identificados:** {sinais}.", icon="🚨")
        return ultima and st.button(BOTAO_SEGUIR, key=f"seguir_{indice}")
    if resposta.tipo == "resposta":
        st.markdown(resposta.texto)
        st.markdown("**Fontes**\n\n" + "\n".join(f"- {_citacao(f)}" for f in resposta.fontes))
        st.caption(
            f"Resposta gerada por IA ({resposta.modelo}) a partir dos manuais. Confira as "
            "fontes. Não substitui avaliação médica. Para achar a UBS mais próxima, use a "
            "página Mapa de UBS."
        )
    elif resposta.tipo == "nao_encontrado":
        st.info(MSG_NAO_ENCONTRADO)
    else:
        st.warning(resposta.texto)
        for f in resposta.fontes:
            trecho = f.texto[:MAX_CARACTERES_TRECHO].rstrip()
            if len(f.texto) > MAX_CARACTERES_TRECHO:
                trecho += "…"
            citado = "\n".join(f"> {_sem_markdown(linha)}" for linha in trecho.splitlines())
            st.markdown(f"{citado}\n\n{_citacao(f)}")
    return False


estilo.aplicar()
st.title("Orientação sobre sintomas")
st.warning(
    "Esta ferramenta não faz diagnóstico e não substitui avaliação médica. Em emergência, "
    "ligue 192 (SAMU) ou procure um pronto-socorro."
)
st.caption(
    "As perguntas são processadas por um serviço externo de inteligência artificial (Groq). "
    "Não escreva nome, CPF, endereço, telefone ou outros dados pessoais. Perguntas que a "
    "ferramenta não consegue responder são registradas, sem identificação, para melhorar a "
    "base de manuais."
)

buscador = data.buscador_rag()
if buscador is None:
    st.info("A base de manuais está indisponível no momento. Tente novamente mais tarde.")
    st.stop()

with st.expander("Manuais consultados"):
    manuais = buscador.trechos.drop_duplicates("manual")
    st.markdown("\n".join(f"- [{m.titulo}]({m.url}) ({m.edicao})" for m in manuais.itertuples()))

conversa: list[dict] = st.session_state.setdefault("conversa", [])
seguir = None
for i, mensagem in enumerate(conversa):
    with st.chat_message(mensagem["papel"]):
        if mensagem["papel"] == "user":
            st.markdown(_sem_markdown(mensagem["texto"]))
        elif _mostrar_resposta(i, mensagem, ultima=i == len(conversa) - 1):
            seguir = mensagem["pergunta"]
if seguir:
    _processar(seguir, ignorar_alarme=True)

pendente = None
if not conversa:
    st.markdown("**Exemplos de perguntas**")
    colunas = st.columns(len(EXEMPLOS))
    for coluna, exemplo in zip(colunas, EXEMPLOS, strict=True):
        if coluna.button(exemplo, key=f"exemplo_{exemplo}", width="stretch"):
            pendente = exemplo
elif st.button("Limpar conversa"):
    st.session_state["conversa"] = []
    st.rerun()

perguntas_feitas = sum(m["papel"] == "user" for m in conversa)
esgotou = perguntas_feitas >= MAX_PERGUNTAS
if esgotou:
    st.info(
        f'Esta conversa chegou ao limite de {MAX_PERGUNTAS} perguntas. Use "Limpar '
        'conversa" para começar outra.'
    )
digitada = st.chat_input(
    "Descreva os sintomas ou pergunte sobre dengue, chikungunya, zika ou febre amarela",
    max_chars=1000,
    disabled=esgotou,
)
pergunta = pendente or (digitada.strip() if digitada else None)
if pergunta and not esgotou:
    with st.chat_message("user"):
        st.markdown(_sem_markdown(pergunta))
    _processar(pergunta)
