"""Avaliação das respostas do chatbot com o Ragas: fidelidade e precisão de contexto.

Manual e opcional (workflow `avaliacao.yml`): roda o chatbot completo (índice do Hub e
LLM) nas perguntas no escopo de `rag/avaliacao.yaml` e mede, com outro LLM como juiz:

- **fidelidade** (`Faithfulness`): fração das afirmações da resposta sustentadas pelos
  trechos enviados ao LLM;
- **precisão de contexto** (`ContextPrecisionWithoutReference`): se os trechos úteis
  vieram nas primeiras posições.

Gasta a cota gratuita do Groq (cerca de 8 mil tokens por minuto e 200 mil por dia em cada
modelo): use poucas perguntas, com espera entre elas, fora do horário de uso do app. O
juiz é o `gpt-oss-20b`, para não disputar a cota do modelo que responde no app.

Uso: `python -m rag.avaliacao_respostas [--perguntas 20] [--sem-publicar]`.
"""

import argparse
import asyncio
import json
import logging
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.config import get_settings
from rag.index import carregar_avaliacao
from rag.resposta import ClienteLLM, responder
from rag.retrieve import Buscador, carregar_buscador

log = logging.getLogger(__name__)

JUIZ = "openai/gpt-oss-20b"
PERGUNTAS_PADRAO = 20
ESPERA_S = 30  # entre perguntas, para caber na cota por minuto


def _media(valores: list[float | None]) -> float | None:
    validos = [v for v in valores if v is not None and not math.isnan(v)]
    return round(sum(validos) / len(validos), 4) if validos else None


async def avaliar(
    perguntas: list[str], buscador: Buscador, llm: ClienteLLM, juiz: Any, espera: float
) -> dict[str, Any]:
    """Responde cada pergunta e pontua as respostas com as métricas do Ragas."""
    from ragas.metrics.collections import ContextPrecisionWithoutReference, Faithfulness

    fidelidade = Faithfulness(llm=juiz)
    precisao = ContextPrecisionWithoutReference(llm=juiz)
    linhas = []
    for i, pergunta in enumerate(perguntas):
        if i:
            await asyncio.sleep(espera)
        resposta = responder(pergunta, [], buscador, llm, ignorar_alarme=True)
        linha: dict[str, Any] = {
            "pergunta": pergunta,
            "tipo": resposta.tipo,
            "motivo": resposta.motivo,
            "fidelidade": None,
            "precisao_contexto": None,
        }
        if resposta.tipo == "resposta":
            busca = buscador.buscar(pergunta)
            contextos = [buscador.trecho(r.posicao)["texto"] for r in busca.resultados]
            try:
                linha["fidelidade"] = (
                    await fidelidade.ascore(
                        user_input=pergunta, response=resposta.texto, retrieved_contexts=contextos
                    )
                ).value
                linha["precisao_contexto"] = (
                    await precisao.ascore(
                        user_input=pergunta, response=resposta.texto, retrieved_contexts=contextos
                    )
                ).value
            except Exception as exc:  # o juiz também tem cota; uma falha não perde as demais
                log.warning("Juiz falhou em %r: %s", pergunta, exc)
        log.info("%s", json.dumps(linha, ensure_ascii=False))
        linhas.append(linha)
    return {
        "avaliado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "modelos": llm.modelos,
        "juiz": JUIZ,
        "perguntas": len(linhas),
        "respondidas": sum(linha["tipo"] == "resposta" for linha in linhas),
        "fidelidade_media": _media([linha["fidelidade"] for linha in linhas]),
        "precisao_contexto_media": _media([linha["precisao_contexto"] for linha in linhas]),
        "por_pergunta": linhas,
    }


def main() -> None:
    """Avalia as respostas e publica `rag/avaliacao_respostas.json` no Hub."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perguntas", type=int, default=PERGUNTAS_PADRAO)
    parser.add_argument("--espera", type=float, default=ESPERA_S, help="Segundos entre perguntas.")
    parser.add_argument("--sem-publicar", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    s = get_settings()
    llm = ClienteLLM.da_configuracao(s)
    if llm is None:
        raise SystemExit("Defina RADAR_LLM_API_KEY para avaliar as respostas.")
    cliente = AsyncOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key.get_secret_value())
    juiz = llm_factory(JUIZ, provider="openai", client=cliente)

    perguntas = [p["pergunta"] for p in carregar_avaliacao().perguntas][: args.perguntas]
    resultado = asyncio.run(avaliar(perguntas, carregar_buscador(s), llm, juiz, args.espera))
    resumo = {k: v for k, v in resultado.items() if k != "por_pergunta"}
    log.info("Resumo: %s", json.dumps(resumo, ensure_ascii=False))
    if args.sem_publicar:
        return

    from etl.publish.hub import publicar_pasta

    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        (pasta / "avaliacao_respostas.json").write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        publicar_pasta(pasta, "rag", "rag: avaliação das respostas com o Ragas")


if __name__ == "__main__":
    main()
