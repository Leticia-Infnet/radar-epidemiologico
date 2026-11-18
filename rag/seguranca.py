"""Filtro de sinais de alarme da dengue, aplicado antes de qualquer busca no corpus."""

import re

from rag.texto import normalizar

# Sinais de alarme do Quadro 1 do manual "Dengue: diagnóstico e manejo clínico" (6ª ed.,
# 2024, p. 14), com variações coloquiais. A hepatomegalia não entra: só o exame físico a
# detecta. Por padrões de texto, com falso positivo aceito: perguntas informativas ("dengue
# causa sangramento?") também disparam o alerta, e o chatbot oferece seguir com a resposta.
# Revisar com profissional de saúde.
SINAIS_DE_ALARME: dict[str, tuple[str, ...]] = {
    "dor abdominal intensa": (
        r"dor (abdominal|na barriga|no estomago) (intensa|forte|muito forte|insuportavel)",
        r"(muita|forte) dor (abdominal|na barriga)",
    ),
    "vômitos persistentes": (
        r"vomit\w* (persistente|sem parar|o tempo todo|varias vezes|muito)",
        r"nao (para|consigo parar) de vomitar",
    ),
    "sangramento": (
        r"sangra\w*",
        r"sangue (na|no|nas|nos) (urina|fezes|gengiva|nariz|vomito)",
        r"(vomit\w*|tossi\w*|cuspi\w*) sangue",
    ),
    "letargia": (
        r"letargi\w*",
        r"(muito )?sonolen\w*",
        r"(confusao mental|desorientad\w*|nao (consegue|consigo) ficar acordad\w*)",
        r"irritabilidade",
        r"(crianca|bebe|nene|filh[oa]).{0,30}(muito irritad\w*|chor\w* sem parar)",
    ),
    "hipotensão postural": (
        r"hipotens\w*",
        r"(tontura|desmai\w*) ao (levantar|ficar de pe)",
        r"pressao (baixa|caindo)",
        r"desmai\w*",
        r"lipotimi\w*",
    ),
    "acúmulo de líquidos ou falta de ar": (
        r"falta de ar",
        r"dificuldade (para|de|pra) respirar",
        r"(barriga|abdome|abdomen).{0,20}(inchad\w*|distendid\w*|cheia de agua)",
        r"(ascite|derrame pleural|derrame pericardico)",
    ),
}

ORIENTACAO_URGENTE = (
    "Os sintomas descritos podem ser sinais de alarme. Procure atendimento médico "
    "imediatamente, em uma UPA ou pronto-socorro, ou ligue 192 (SAMU)."
)


def sinais_detectados(mensagem: str) -> list[str]:
    """Devolve os sinais de alarme encontrados na mensagem do usuário."""
    texto = normalizar(mensagem)
    return [
        sinal
        for sinal, padroes in SINAIS_DE_ALARME.items()
        if any(re.search(p, texto) for p in padroes)
    ]
