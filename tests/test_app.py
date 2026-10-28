from pathlib import Path

from streamlit.testing.v1 import AppTest

MAIN = Path(__file__).parents[1] / "app" / "main.py"


def test_pagina_inicial_renderiza():
    app = AppTest.from_file(str(MAIN)).run()
    assert not app.exception
    assert app.title[0].value == "Radar Epidemiológico"


def test_paginas_renderizam():
    for pagina in sorted((MAIN.parent / "pages").glob("*.py")):
        app = AppTest.from_file(str(pagina)).run()
        assert not app.exception, pagina.name
