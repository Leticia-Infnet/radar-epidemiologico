"""Leitura das partições `doenca=<agravo>/uf=<código>/casos.parquet` do Hub.

Em produção (2026-11-18), o pyarrow não confirmou que o caminho era um arquivo na primeira
consulta ao Hub e o leu como dataset: descobriu `doenca` e `uf` no caminho como partições
Hive, e a `doenca` do caminho (dicionário) colidiu com a do arquivo (texto), com
ArrowTypeError. Os testes simulam a falha com um sistema de arquivos em memória cuja
primeira consulta de metadados falha.
"""

import uuid

import fsspec
import pandas as pd
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from app import data
from ml import dados

CASOS = pd.DataFrame(
    {
        "doenca": ["dengue"] * 4,
        "codigo_ibge6": ["260960", "260960", "261160", "261160"],
        "uf_codigo": ["26"] * 4,
        "ano_epi": [2026] * 4,
        "semana_epi": [1, 2, 1, 2],
        "inicio_semana": pd.to_datetime(["2026-01-04", "2026-01-11"] * 2).date,
        "casos_provaveis": [10, 12, 3, 4],
        "obitos": [0, 1, 0, 0],
        "semana_incompleta": [False, True, False, True],
    }
)


class HubInstavel(MemoryFileSystem):
    """Memória com a primeira consulta de metadados de cada arquivo falhando."""

    protocol = "hubinstavel"
    falhas: set[str] = set()

    def info(self, path, **kwargs):
        caminho = self._strip_protocol(path)
        if caminho.endswith(".parquet") and caminho not in self.falhas:
            self.falhas.add(caminho)
            raise FileNotFoundError(caminho)
        return super().info(path, **kwargs)


fsspec.register_implementation("hubinstavel", HubInstavel, clobber=True)


@pytest.fixture
def url_hub():
    """Grava a partição e devolve a função que monta a URL de um caminho do dataset."""
    raiz = f"hubinstavel://{uuid.uuid4().hex}"
    with fsspec.open(f"{raiz}/arboviroses/doenca=dengue/uf=26/casos.parquet", "wb") as f:
        CASOS.to_parquet(f, index=False)
    return lambda caminho: f"{raiz}/{caminho}"


def test_a_falha_de_metadados_reproduz_o_erro_de_producao(url_hub):
    with pytest.raises(TypeError, match="doenca"):
        pd.read_parquet(url_hub("arboviroses/doenca=dengue/uf=26/casos.parquet"))


def test_casos_uf_le_a_particao_mesmo_com_a_falha(url_hub, monkeypatch):
    monkeypatch.setattr(data, "_url_hub", url_hub)
    data.casos_uf.clear()
    tabela = data.casos_uf("dengue", "26")
    data.casos_uf.clear()
    assert len(tabela) == 4
    assert "doenca" not in tabela.columns and "uf" not in tabela.columns
    assert tabela["casos_provaveis"].sum() == 29


def test_treino_le_a_particao_mesmo_com_a_falha(url_hub, monkeypatch):
    monkeypatch.setattr(dados, "_url", url_hub)
    monkeypatch.setattr(dados, "MEDIA_MINIMA", 5)
    elegiveis = dados._municipios_elegiveis("dengue", "26", completas=2)
    assert list(elegiveis.columns) == ["260960"]
