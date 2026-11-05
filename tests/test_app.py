from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_pagina_dashboard import sem_rede  # noqa: F401 (fixture)

MAIN = Path(__file__).parents[1] / "app" / "main.py"


def test_pagina_inicial_renderiza():
    app = AppTest.from_file(str(MAIN)).run()
    assert not app.exception
    assert app.title[0].value == "Radar Epidemiológico"


def test_paginas_renderizam(sem_rede):  # noqa: F811
    """Toda página abre sem erro; o dashboard lê o Hub ao abrir, então ele vai simulado."""
    for pagina in sorted((MAIN.parent / "pages").glob("*.py")):
        app = AppTest.from_file(str(pagina)).run()
        assert not app.exception, pagina.name
