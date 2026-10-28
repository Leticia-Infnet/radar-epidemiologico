"""Dicionário de municípios a partir da API de Localidades do IBGE."""

import httpx
import pandas as pd

from common.config import Settings, get_settings


def municipios(settings: Settings | None = None) -> pd.DataFrame:
    """Devolve os municípios com código IBGE de 7 e de 6 dígitos, nome e UF.

    O código de 6 dígitos (sem dígito verificador) é a chave usada pela DEMAS.
    """
    settings = settings or get_settings()
    resposta = httpx.get(
        settings.ibge_municipios_url,
        headers={"User-Agent": settings.user_agent},
        timeout=60,
    )
    resposta.raise_for_status()

    linhas = []
    for m in resposta.json():
        # Alguns municípios vêm sem microrregião; a UF fica sempre na região imediata.
        uf = m["regiao-imediata"]["regiao-intermediaria"]["UF"]
        codigo7 = str(m["id"])
        linhas.append(
            {
                "codigo_ibge7": codigo7,
                "codigo_ibge6": codigo7[:6],
                "nome": m["nome"],
                "uf_sigla": uf["sigla"],
                "uf_codigo": str(uf["id"]),
                "regiao": uf["regiao"]["nome"],
            }
        )
    return pd.DataFrame(linhas)
