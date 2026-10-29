"""Normalização do cadastro de UBS da DEMAS para o contrato `UBS`."""

from typing import Any

import polars as pl

from etl.transform.normalize import (
    ausente,
    codigo_cnes,
    codigo_ibge6,
    coordenada,
    dentro_do_brasil,
)

# Com até 2 casas decimais, a coordenada tem precisão de cerca de 1 km.
CASAS_DECIMAIS_PRECISAS = 3

MOTIVOS_DE_DESCARTE = (
    "CNES inválido",
    "código IBGE inválido",
    "sem coordenada",
    "fora do Brasil",
    "CNES repetido",
)


def _casas_decimais(valor: Any) -> int:
    texto = str(valor).strip().replace(",", ".")
    return len(texto.split(".")[1]) if "." in texto else 0


def _texto(valor: Any) -> str | None:
    return None if ausente(valor) else str(valor).strip()


def normalizar_ubs(registros: list[dict[str, Any]]) -> tuple[pl.DataFrame, dict[str, int]]:
    """Normaliza o cadastro, descarta UBS sem localização utilizável e marca as aproximadas.

    Os motivos de descarte são avaliados na ordem de `MOTIVOS_DE_DESCARTE`; cada registro
    conta só no primeiro motivo em que falha. Devolve a tabela e a contagem por motivo.
    """
    descartes = dict.fromkeys(MOTIVOS_DE_DESCARTE, 0)
    vistos: set[str] = set()
    linhas = []
    for registro in registros:
        cnes = codigo_cnes(registro.get("cnes"))
        ibge = codigo_ibge6(registro.get("ibge"))
        lat = coordenada(registro.get("latitude"))
        lon = coordenada(registro.get("longitude"))
        if cnes is None:
            motivo = "CNES inválido"
        elif ibge is None:
            motivo = "código IBGE inválido"
        elif lat is None or lon is None:
            motivo = "sem coordenada"
        elif not dentro_do_brasil(lat, lon):
            motivo = "fora do Brasil"
        elif cnes in vistos:
            motivo = "CNES repetido"
        else:
            motivo = None

        if motivo:
            descartes[motivo] += 1
            continue
        vistos.add(cnes)
        precisao = min(
            _casas_decimais(registro["latitude"]), _casas_decimais(registro["longitude"])
        )
        linhas.append(
            {
                "cnes": cnes,
                "nome": _texto(registro.get("nome")) or "",
                "codigo_ibge6": ibge,
                "uf_codigo": ibge[:2],
                "logradouro": _texto(registro.get("logradouro")),
                "bairro": _texto(registro.get("bairro")),
                "latitude": lat,
                "longitude": lon,
                "coordenada_aproximada": precisao < CASAS_DECIMAIS_PRECISAS,
            }
        )

    esquema = {
        "cnes": pl.String,
        "nome": pl.String,
        "codigo_ibge6": pl.String,
        "uf_codigo": pl.String,
        "logradouro": pl.String,
        "bairro": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "coordenada_aproximada": pl.Boolean,
    }
    return pl.DataFrame(linhas, schema=esquema), descartes
