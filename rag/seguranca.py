"""Filtro de sinais de alarme da dengue, aplicado antes de qualquer busca no corpus."""

import re
import unicodedata

# Sinais de alarme listados no AGENTS.md, seção 7.4, com variações coloquiais.
# Primeira versão por padrões de texto; revisar com profissional de saúde.
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
    ),
    "hipotensão postural": (
        r"hipotens\w*",
        r"(tontura|desmai\w*) ao (levantar|ficar de pe)",
        r"pressao (baixa|caindo)",
    ),
}

ORIENTACAO_URGENTE = (
    "Os sintomas descritos podem ser sinais de alarme. Procure atendimento médico "
    "imediatamente, em uma UPA ou pronto-socorro, ou ligue 192 (SAMU)."
)


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento.lower())


def sinais_detectados(mensagem: str) -> list[str]:
    """Devolve os sinais de alarme encontrados na mensagem do usuário."""
    texto = _normalizar(mensagem)
    return [
        sinal
        for sinal, padroes in SINAIS_DE_ALARME.items()
        if any(re.search(p, texto) for p in padroes)
    ]
