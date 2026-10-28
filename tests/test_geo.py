import pandas as pd
import pytest

from app.geo import construir_indice, ubs_mais_proximas

UBS = pd.DataFrame(
    {
        "cnes": ["0000001", "0000002", "0000003"],
        "latitude": [-23.5505, -22.9068, -8.0476],
        "longitude": [-46.6333, -43.1729, -34.8770],
    }
)


def test_ubs_mais_proxima_e_distancia():
    indice = construir_indice(UBS)
    # Ponto em Campinas: São Paulo deve vir antes do Rio.
    resultado = ubs_mais_proximas(indice, UBS, -22.9056, -47.0608, k=2)
    assert list(resultado["cnes"]) == ["0000001", "0000002"]
    assert resultado["distancia_km"].iloc[0] == pytest.approx(84, abs=5)


def test_k_fora_do_intervalo():
    with pytest.raises(ValueError):
        ubs_mais_proximas(construir_indice(UBS), UBS, 0, 0, k=6)
