from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import data, geo
from app.geo import CepNaoEncontrado, Localizacao, construir_indice
from tests.test_geo import UBS

PAGINA = str(Path(__file__).parents[1] / "app" / "pages" / "3_mapa.py")


@pytest.fixture
def sem_rede(monkeypatch):
    """Substitui Hub, APIs de CEP e CNES por dados locais."""
    monkeypatch.setattr(data, "indice_ubs", lambda: (construir_indice(UBS), UBS))
    monkeypatch.setattr(
        geo,
        "detalhar_estabelecimento",
        lambda cnes: {"nome": f"Detalhe {cnes}", "telefone": "(81) 0000-0000"},
    )


def _buscar(cep: str, quantidade: int = 2) -> AppTest:
    app = AppTest.from_file(PAGINA, default_timeout=30)
    app.run()
    app.text_input[0].input(cep)
    app.slider[0].set_value(quantidade)
    app.button[0].click()
    return app.run()


def test_busca_mostra_unidades(sem_rede, monkeypatch):
    origem = Localizacao(-8.05, -34.88, "BrasilAPI", "Rua Teste, Recife, PE")
    monkeypatch.setattr(geo, "geocodificar_cep", lambda cep: origem)
    app = _buscar("50050-000")
    assert not app.exception
    textos = " ".join(m.value for m in app.markdown)
    assert "1. Detalhe 0000003" in textos  # Recife é a mais próxima
    assert "Rua Teste, Recife, PE" in textos


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
