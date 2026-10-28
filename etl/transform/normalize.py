"""Normalizações das armadilhas conhecidas da DEMAS (AGENTS.md, seção 8)."""

import math
from typing import Any

# Bounding box do Brasil continental e ilhas oceânicas.
LAT_MIN, LAT_MAX = -34.0, 6.0
LON_MIN, LON_MAX = -74.0, -34.0

# A DEMAS serializa ausência de várias formas, inclusive a string "nan".
_VALORES_AUSENTES = {"", "nan", "none", "null", "nat"}


def ausente(valor: Any) -> bool:
    """Indica se o valor representa dado ausente em qualquer das formas da DEMAS."""
    if valor is None:
        return True
    if isinstance(valor, float):
        return math.isnan(valor)
    return isinstance(valor, str) and valor.strip().lower() in _VALORES_AUSENTES


def coordenada(valor: Any) -> float | None:
    """Converte coordenada com vírgula decimal (`"-8,21811"`) para float."""
    if ausente(valor):
        return None
    try:
        return float(str(valor).strip().replace(",", "."))
    except ValueError:
        return None


def dentro_do_brasil(lat: float | None, lon: float | None) -> bool:
    """Verifica se o par lat/lon cai na bounding box do Brasil."""
    if lat is None or lon is None:
        return False
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def codigo_ibge6(valor: Any) -> str | None:
    """Padroniza código de município para 6 dígitos, o formato da DEMAS.

    Aceita 6 ou 7 dígitos, como texto ou número (a DEMAS às vezes devolve inteiro).
    """
    if ausente(valor):
        return None
    texto = str(valor).strip()
    if texto.endswith(".0"):
        texto = texto[:-2]
    if not texto.isdigit() or len(texto) not in (6, 7):
        return None
    return texto[:6]


def codigo_cnes(valor: Any) -> str | None:
    """Padroniza o CNES como texto de 7 dígitos, preservando zeros à esquerda."""
    if ausente(valor):
        return None
    texto = str(valor).strip()
    if texto.endswith(".0"):
        texto = texto[:-2]
    if not texto.isdigit() or len(texto) > 7:
        return None
    return texto.zfill(7)
