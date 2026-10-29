from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import data, geo
from app.geo import CepNaoEncontrado, Localizacao, Rota, construir_indice
from tests.test_geo import UBS

PAGINA = str(Path(__file__).parents[1] / "app" / "pages" / "3_mapa.py")
RECIFE = Localizacao(-8.05, -34.88, "BrasilAPI", "Rua Teste, Recife, PE")


@pytest.fixture
def sem_rede(monkeypatch):
    """Substitui Hub, CNES e o servidor de rotas por dados locais."""
    monkeypatch.setattr(data, "indice_ubs", lambda: (construir_indice(UBS), UBS))
    monkeypatch.setattr(
        geo,
        "detalhar_estabelecimento",
        lambda cnes: {"nome": f"Detalhe {cnes}", "telefone": "(81) 0000-0000"},
    )
    # Recife (0000003) fica a 4 km a pé; o Rio (0000002), a 900 m: a ordem deve inverter.
    a_pe = {"0000003": (4000.0, 3000.0), "0000002": (900.0, 700.0), "0000001": None}

    def distancias(origem, destinos):
        por_ponto = {(u.latitude, u.longitude): a_pe[u.cnes] for u in UBS.itertuples()}
        return [por_ponto[d] for d in destinos]

    monkeypatch.setattr(geo, "distancias_a_pe", distancias)
    monkeypatch.setattr(
        geo,
        "rota_a_pe",
        lambda o, lat, lon: Rota(900.0, 700.0, ((o.latitude, o.longitude), (lat, lon))),
    )


def _buscar(cep: str, quantidade: int = 2) -> AppTest:
    app = AppTest.from_file(PAGINA, default_timeout=30)
    app.run()
    app.text_input[0].input(cep)
    app.slider[0].set_value(quantidade)
    app.button[0].click()
    return app.run()


def test_busca_ordena_a_pe_e_mostra_rota(sem_rede, monkeypatch):
    monkeypatch.setattr(geo, "geocodificar_cep", lambda cep: RECIFE)
    app = _buscar("50050-000", quantidade=2)
    assert not app.exception
    textos = " ".join(m.value for m in app.markdown)
    assert "1. Detalhe 0000002" in textos  # mais perto a pé, embora mais longe em linha reta
    assert "Rua Teste, Recife, PE" in textos
    assert app.radio[0].value == "0000002"
    legendas = " ".join(c.value for c in app.caption)
    assert "Caminho a pé: 900 m, cerca de 12 min" in legendas


def test_trocar_destino_mantem_resultado(sem_rede, monkeypatch):
    monkeypatch.setattr(geo, "geocodificar_cep", lambda cep: RECIFE)
    app = _buscar("50050-001", quantidade=2)
    app.radio[0].set_value("0000003")
    app.run()
    assert not app.exception
    assert app.radio[0].value == "0000003"
    assert len([m for m in app.markdown if m.value.startswith("**1.")]) == 1


def test_cep_invalido_mostra_erro(sem_rede):
    app = _buscar("123")
    assert not app.exception
    assert "8 dígitos" in app.error[0].value


def test_cep_nao_encontrado_mostra_erro(sem_rede, monkeypatch):
    def nao_acha(cep):
        raise CepNaoEncontrado(cep)

    monkeypatch.setattr(geo, "geocodificar_cep", nao_acha)
    app = _buscar("69900-001")
    assert not app.exception
    assert "Não encontramos" in app.error[0].value
