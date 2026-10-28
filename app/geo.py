"""Geocodificação de CEP e busca das UBS mais próximas."""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

RAIO_TERRA_KM = 6371.0088


def construir_indice(ubs: pd.DataFrame) -> BallTree:
    """Índice haversine sobre as coordenadas das UBS, em radianos."""
    return BallTree(np.radians(ubs[["latitude", "longitude"]].to_numpy()), metric="haversine")


def ubs_mais_proximas(
    indice: BallTree, ubs: pd.DataFrame, lat: float, lon: float, k: int = 5
) -> pd.DataFrame:
    """Devolve as k UBS mais próximas do ponto, com a distância em linha reta (km)."""
    if not 1 <= k <= 5:
        raise ValueError("k deve estar entre 1 e 5.")
    distancias, posicoes = indice.query(np.radians([[lat, lon]]), k=k)
    resultado = ubs.iloc[posicoes[0]].copy()
    resultado["distancia_km"] = distancias[0] * RAIO_TERRA_KM
    return resultado.reset_index(drop=True)


def geocodificar_cep(cep: str) -> tuple[float, float] | None:
    """Converte CEP em lat/lon: BrasilAPI, depois AwesomeAPI, depois ViaCEP + Nominatim."""
    raise NotImplementedError
