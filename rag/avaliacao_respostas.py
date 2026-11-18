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
from rag.resposta import ClienteLLM, formatar_trecho, responder
from rag.retrieve import Buscador, carregar_buscador

log = logging.getLogger(__name__)

JUIZ = "openai/gpt-oss-20b"
# Com a configuração padrão do Ragas, o raciocínio do gpt-oss gasta os tokens e o Groq
# devolve "Failed to generate JSON" (13 de 14 respostas na primeira execução, 2026-11-18).
# Modo JSON, mais tokens e raciocínio baixo resolveram no mesmo caso.
OPCOES_DO_JUIZ = {"max_tokens": 4096, "reasoning_effort": "low"}
MINIMO_AVALIADAS = 0.5  # fração das respostas que o juiz precisa pontuar
PERGUNTAS_PADRAO = 20
ESPERA_S = 30  # entre perguntas, para caber na cota por minuto
# O juiz faz várias chamadas longas por pergunta e esbarra no limite de tokens por minuto
# do Groq (8 de 20 perguntas sem nota em 2026-11-18): esperar e tentar de novo. O limite
# diário não volta em um minuto, então nele não há nova tentativa.
TENTATIVAS_429 = 3
ESPERA_429_S = 60


def limite_por_minuto(exc: Exception) -> bool:
    """O erro é o 429 do limite por minuto (passa sozinho), e não o diário."""
    texto = str(exc)
    return ("429" in texto or "Rate limit" in texto) and "per day" not in texto


async def pontuar(
    metrica: Any,
    tentativas: int = TENTATIVAS_429,
    espera: float = ESPERA_429_S,
    **argumentos: Any,
) -> float | None:
    """Nota de uma métrica; tenta de novo no limite por minuto e devolve None se falhar."""
    for tentativa in range(1, tentativas + 1):
        try:
            return (await metrica.ascore(**argumentos)).value
        except Exception as exc:  # o juiz também tem cota; uma falha não perde as demais
            if limite_por_minuto(exc) and tentativa < tentativas:
                log.info("Limite por minuto do juiz; nova tentativa em %.0f s.", espera)
                await asyncio.sleep(espera)
                continue
            log.warning("Juiz falhou em %r: %s", argumentos.get("user_input"), exc)
            return None
    return None


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
            # Os trechos como o chatbot os recebe, com manual e seção: às vezes a informação
            # está no título da seção (um aviso em destaque que o Docling leu como título).
            contextos = [
                formatar_trecho(n, buscador.trecho(r.posicao))
                for n, r in enumerate(busca.resultados, start=1)
            ]
            argumentos = {
                "user_input": pergunta,
                "response": resposta.texto,
                "retrieved_contexts": contextos,
            }
            linha["fidelidade"] = await pontuar(fidelidade, **argumentos)
            linha["precisao_contexto"] = await pontuar(precisao, **argumentos)
        log.info("%s", json.dumps(linha, ensure_ascii=False))
        linhas.append(linha)
    return {
        "avaliado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "modelos": llm.modelos,
        "juiz": JUIZ,
        "perguntas": len(linhas),
        "respondidas": sum(linha["tipo"] == "resposta" for linha in linhas),
        "avaliadas": sum(linha["fidelidade"] is not None for linha in linhas),
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

    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    s = get_settings()
    llm = ClienteLLM.da_configuracao(s)
    if llm is None:
        raise SystemExit("Defina RADAR_LLM_API_KEY para avaliar as respostas.")
    cliente = AsyncOpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key.get_secret_value())
    juiz = llm_factory(
        JUIZ, provider="openai", client=cliente, mode=instructor.Mode.JSON, **OPCOES_DO_JUIZ
    )

    perguntas = [p["pergunta"] for p in carregar_avaliacao().perguntas][: args.perguntas]
    resultado = asyncio.run(avaliar(perguntas, carregar_buscador(s), llm, juiz, args.espera))
    resumo = {k: v for k, v in resultado.items() if k != "por_pergunta"}
    log.info("Resumo: %s", json.dumps(resumo, ensure_ascii=False))
    if resultado["avaliadas"] < MINIMO_AVALIADAS * max(resultado["respondidas"], 1):
        raise SystemExit(
            f"O juiz só avaliou {resultado['avaliadas']} de {resultado['respondidas']} "
            "respostas: o resultado não é confiável e não será publicado."
        )
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
