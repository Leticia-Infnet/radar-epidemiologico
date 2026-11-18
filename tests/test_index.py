import math

import pandas as pd
import pytest

from rag.index import (
    Avaliacao,
    IndiceReprovado,
    avaliar,
    calibrar_limiar,
    carregar_avaliacao,
    conferir_qualidade,
    relevante,
)
from tests.test_retrieve import buscador_sintetico


def test_relevante_exige_manual_e_todos_os_termos():
    trecho = {"manual": "dengue", "secao": "Sinais de alarme", "texto": "Dor abdominal intensa."}
    assert relevante(trecho, {"manuais": ["dengue"], "termos": ["sinais de alarme", "dor"]})
    assert not relevante(trecho, {"manuais": ["chik"], "termos": ["dor"]})
    assert not relevante(trecho, {"manuais": ["dengue"], "termos": ["vomitos"]})


def test_avaliar_mede_acerto_e_mrr():
    avaliacao = Avaliacao(
        perguntas=[
            {
                "pergunta": "sinais de alarme da dengue",
                "manuais": ["dengue"],
                "termos": ["letargia"],
            },
            {"pergunta": "vacina da febre amarela", "manuais": ["febre"], "termos": ["vacina"]},
        ],
        fora_do_escopo=["receita de bolo de cenoura"],
    )
    metricas = avaliar(buscador_sintetico(), avaliacao)
    assert metricas["hit_at_k"] == 1.0
    assert metricas["mrr"] == 1.0
    assert metricas["falhas"] == []
    assert len(metricas["sims_fora"]) == 1


def test_avaliar_recusa_pergunta_sem_gabarito_no_corpus():
    avaliacao = Avaliacao([{"pergunta": "x", "manuais": ["dengue"], "termos": ["inexistente"]}], [])
    with pytest.raises(ValueError, match="sem nenhum trecho relevante"):
        avaliar(buscador_sintetico(), avaliacao)


def test_limiar_no_meio_da_folga():
    escopo = [0.87, 0.88, 0.89, 0.90, 0.91] * 4
    fora = [0.77, 0.80, 0.84]
    limiar, rejeicao = calibrar_limiar(escopo, fora)
    assert 0.84 < limiar < 0.87
    assert rejeicao == 1.0


def test_limiar_sem_folga_usa_o_quantil():
    escopo = [0.80 + i / 100 for i in range(10)]
    limiar, rejeicao = calibrar_limiar(escopo, [0.85])
    assert limiar == pytest.approx(0.804, abs=0.001)
    assert rejeicao == 0.0


def _metricas(**mudancas):
    base = {
        "k": 4,
        "n_escopo": 30,
        "n_fora": 10,
        "hit_at_k": 0.9,
        "rejeicao_fora": 0.8,
        "limiar": 0.85,
        "falhas": [],
    }
    return {**base, **mudancas}


def test_portao_de_qualidade():
    trechos = pd.DataFrame({"manual": ["a"] * 20 + ["b"] * 20})
    conferir_qualidade(trechos, _metricas(), ["a", "b"])
    with pytest.raises(IndiceReprovado, match="b: 3 trechos"):
        conferir_qualidade(
            pd.DataFrame({"manual": ["a"] * 20 + ["b"] * 3}), _metricas(), ["a", "b"]
        )
    with pytest.raises(IndiceReprovado, match="Acerto"):
        conferir_qualidade(trechos, _metricas(hit_at_k=0.5), ["a"])
    with pytest.raises(IndiceReprovado, match="fora do escopo"):
        conferir_qualidade(trechos, _metricas(rejeicao_fora=0.2), ["a"])
    assert math.isnan(calibrar_limiar([0.9], [])[1])


def test_avaliacao_versionada_e_valida():
    """O arquivo do repositório tem perguntas suficientes e campos completos."""
    avaliacao = carregar_avaliacao()
    assert len(avaliacao.perguntas) >= 30
    assert len(avaliacao.fora_do_escopo) >= 10
    for p in avaliacao.perguntas:
        assert p["pergunta"] and p["manuais"] and p["termos"]
    so_dengue = carregar_avaliacao(ids=["dengue-manejo-clinico"])
    assert all(p["manuais"] == ["dengue-manejo-clinico"] for p in so_dengue.perguntas)
