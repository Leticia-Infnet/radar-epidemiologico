"""Registro das perguntas que o chatbot não soube responder, para melhorar o corpus.

As perguntas vão para um dataset **privado** e separado no Hub
(`RADAR_HF_CONSULTAS_REPO`), pelo `CommitScheduler` do `huggingface_hub`: o app grava em
JSONL num diretório temporário do servidor (disco efêmero do Community Cloud, não a
máquina do usuário) e o scheduler envia a cada 10 minutos e ao encerrar.

Só a pergunta (com CPF, e-mail, telefone e CEP mascarados), a data, o motivo e a
similaridade. Nada que identifique a sessão ou o usuário.
"""

import json
import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from common.config import Settings, get_settings

log = logging.getLogger(__name__)

MAX_CARACTERES = 1000
INTERVALO_MINUTOS = 10

_MASCARAS = [
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b"), "[EMAIL]"),
    (re.compile(r"\b\d{5}-?\d{3}\b"), "[CEP]"),
    (re.compile(r"(\(?\b\d{2}\)?\s?)?\b9?\d{4}[-\s]?\d{4}\b"), "[TELEFONE]"),
]


def mascarar(texto: str) -> str:
    """Troca CPF, e-mail, CEP e telefone por marcadores (nessa ordem)."""
    for padrao, marcador in _MASCARAS:
        texto = padrao.sub(marcador, texto)
    return texto


class Registrador(Protocol):
    def registrar(self, pergunta: str, motivo: str, similaridade: float | None) -> None: ...


class Registro:
    """Grava as perguntas num JSONL próprio deste processo, enviado pelo scheduler."""

    def __init__(self, scheduler: Any, pasta: Path) -> None:
        self._scheduler = scheduler
        # Um arquivo por processo: réplicas e reinícios não sobrescrevem uns aos outros.
        self.arquivo = pasta / f"{uuid4()}.jsonl"

    def registrar(self, pergunta: str, motivo: str, similaridade: float | None) -> None:
        linha = {
            "quando": datetime.now(UTC).isoformat(timespec="seconds"),
            "pergunta": mascarar(pergunta)[:MAX_CARACTERES],
            "motivo": motivo,
            "similaridade": similaridade,
        }
        with self._scheduler.lock, self.arquivo.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")


class RegistroNulo:
    """Sem token configurado (desenvolvimento local, testes): só o log."""

    def registrar(self, pergunta: str, motivo: str, similaridade: float | None) -> None:
        log.info("Pergunta sem resposta (%s, similaridade %s).", motivo, similaridade)


def criar_registro(settings: Settings | None = None) -> Registro | RegistroNulo:
    """Registro ligado ao dataset privado, ou o nulo se faltar o token."""
    s = settings or get_settings()
    if s.hf_consultas_token is None:
        return RegistroNulo()
    from huggingface_hub import CommitScheduler

    pasta = Path(tempfile.mkdtemp(prefix="consultas-"))
    scheduler = CommitScheduler(
        repo_id=s.hf_consultas_repo,
        repo_type="dataset",
        folder_path=pasta,
        path_in_repo="consultas",
        every=INTERVALO_MINUTOS,
        token=s.hf_consultas_token.get_secret_value(),
        private=True,
        allow_patterns=["*.jsonl"],
    )
    return Registro(scheduler, pasta)
