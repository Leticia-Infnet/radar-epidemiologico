"""Ponto de entrada do ETL, chamado pelo workflow `etl.yml`.

Uso: `uv run python -m etl.pipeline {municipios,ubs,arboviroses}`
"""

import argparse


def executar_municipios() -> None:
    """Materializa o dicionário de municípios do IBGE e publica no Hub."""
    raise NotImplementedError


def executar_ubs() -> None:
    """Extrai, normaliza, valida e publica o cadastro de UBS."""
    raise NotImplementedError


def executar_arboviroses() -> None:
    """Extrai o recorte definido, agrega por semana epidemiológica e publica no Hub."""
    raise NotImplementedError


ETAPAS = {
    "municipios": executar_municipios,
    "ubs": executar_ubs,
    "arboviroses": executar_arboviroses,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapas", nargs="+", choices=ETAPAS)
    for etapa in parser.parse_args().etapas:
        ETAPAS[etapa]()


if __name__ == "__main__":
    main()
