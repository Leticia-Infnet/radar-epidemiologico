"""Agregação dos vínculos do CNES em profissionais por UBS e ocupação (CBO).

Cada linha do arquivo PF é um vínculo: o mesmo profissional aparece mais de uma vez quando
tem mais de um vínculo na unidade. A contagem é de profissionais distintos por unidade e
ocupação. Nomes, CPF e CNS servem só para essa contagem e não saem desta etapa.
"""

from collections import defaultdict
from collections.abc import Iterable

import polars as pl

from etl.transform.cbo import OCUPACOES_COMPLEMENTARES, grupo_da_ocupacao
from etl.transform.normalize import codigo_cnes

# Campos lidos do arquivo PF, na ordem em que `agregar_profissionais` os recebe.
CAMPOS_PF = ("CNES", "CBO", "CNS_PROF", "CPF_PROF")

ESQUEMA = {
    "cnes": pl.String,
    "grupo": pl.String,
    "cbo": pl.String,
    "ocupacao": pl.String,
    "profissionais": pl.Int64,
    "competencia": pl.String,
}


def competencia_iso(aamm: str) -> str:
    """`2609` → `2026-09`."""
    return f"20{aamm[:2]}-{aamm[2:]}"


def _sem_nome(cbo: str, familias: dict[str, str]) -> str:
    return f"{familias.get(cbo[:4], 'Ocupação')} (CBO {cbo})"


def agregar_profissionais(
    vinculos: Iterable[tuple[str, ...]],
    cnes_validos: set[str],
    ocupacoes: dict[str, str],
    competencia: str,
    familias: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Conta profissionais distintos por UBS e ocupação de atendimento.

    `vinculos` traz tuplas com os campos de `CAMPOS_PF`. O profissional é identificado
    pelo CNS e, na falta dele, pelo CPF (que o DATASUS publica codificado). Vínculos de
    unidades fora de `cnes_validos`, de ocupações fora de `GRUPOS` ou sem identificação
    são ignorados. Ocupação ausente da tabela CBO leva o nome da família (`familias`, por
    código de 4 posições) e o código, como `Enfermeiros (CBO 223580)`.
    """
    ocupacoes = {**OCUPACOES_COMPLEMENTARES, **ocupacoes}
    pessoas: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cnes, cbo, cns, cpf in vinculos:
        cnes = codigo_cnes(cnes)
        if cnes not in cnes_validos or grupo_da_ocupacao(cbo) is None:
            continue
        pessoa = cns or cpf
        if pessoa:
            pessoas[(cnes, cbo)].add(pessoa)

    linhas = [
        {
            "cnes": cnes,
            "grupo": grupo_da_ocupacao(cbo),
            "cbo": cbo,
            "ocupacao": ocupacoes.get(cbo) or _sem_nome(cbo, familias or {}),
            "profissionais": len(ids),
            "competencia": competencia,
        }
        for (cnes, cbo), ids in pessoas.items()
    ]
    return pl.DataFrame(linhas, schema=ESQUEMA).sort(["cnes", "grupo", "cbo"])
