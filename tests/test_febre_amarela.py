from datetime import date

import pytest

from etl import pipeline
from etl.transform.febre_amarela import normalizar_febre_amarela, periodo_de_monitoramento
from etl.transform.schemas import FebreAmarelaMensal


def _caso(**campos):
    base = {
        "mun_lpi": "ALTO ALEGRE",
        "cod_mun_lpi": 140005.0,
        "cod_uf_lpi": 14.0,
        "uf_lpi": "RR",
        "dt_is": "29/11/1994",
        "ano_is": 1994.0,
        "mes_is": 11.0,
        "monitoramento_is": "1994/1995",
        "obito": "SIM",
        "sexo": "M",
        "idade": 30.0,
    }
    return {**base, **campos}


@pytest.mark.parametrize(
    ("dia", "periodo"),
    [
        (date(2024, 7, 1), "2024/2025"),
        (date(2025, 6, 30), "2024/2025"),
        (date(2025, 1, 5), "2024/2025"),
    ],
)
def test_periodo_de_monitoramento_vai_de_julho_a_junho(dia, periodo):
    assert periodo_de_monitoramento(dia) == periodo


def test_normalizar_febre_amarela():
    registros = [
        _caso(),
        _caso(),  # linha idêntica repetida pela DEMAS
        _caso(dt_is="02/12/1994", obito="NÃO", idade=40.0),
        # Sem data: período e mês vêm dos outros campos; município ausente.
        _caso(dt_is=None, cod_mun_lpi=float("nan"), mes_is=3.0, ano_is=1995.0, obito="IGN"),
        _caso(dt_is=None, monitoramento_is=None),  # sem como saber o período
    ]
    tabela, descartes, mais_recente = normalizar_febre_amarela(registros)
    FebreAmarelaMensal.validate(tabela)
    assert descartes == {"registro repetido": 1, "sem data nem período": 1}
    assert mais_recente == date(1994, 12, 2)
    assert tabela.rows() == [
        ("1994/1995", 1994, 1994, 11, "140005", "14", 1, 1),
        ("1994/1995", 1994, 1994, 12, "140005", "14", 1, 0),
        ("1994/1995", 1994, 1995, 3, None, "14", 1, 0),
    ]
    # Idade e sexo não saem desta etapa.
    assert not {"idade", "sexo"} & set(tabela.columns)


def test_febre_amarela_abaixo_do_piso_nao_publica(monkeypatch):
    async def poucos():
        return [_caso()]

    publicacoes = []
    monkeypatch.setattr(pipeline, "_baixar_febre_amarela", poucos)
    monkeypatch.setattr(pipeline, "publicar_pasta", lambda *a, **k: publicacoes.append(a))
    with pytest.raises(pipeline.ExtracaoIncompleta, match="febre_amarela"):
        pipeline.executar_febre_amarela(publicar=True)
    assert publicacoes == []
