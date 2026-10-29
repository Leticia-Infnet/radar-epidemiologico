import pytest

from etl.transform.schemas import UBS
from etl.transform.ubs import normalizar_ubs


def _ubs(**campos):
    base = {
        "ibge": "260290",
        "bairro": "PONTE DOS CARVALHOS",
        "logradouro": "RUA DO CEMITERIO",
        "cnes": "0000302",
        "uf": "26",
        "latitude": "-8,21811",
        "longitude": "-35,22944",
        "nome": "USF SANTO ESTEVAO",
    }
    return {**base, **campos}


def test_normaliza_registro_valido():
    tabela, descartes = normalizar_ubs([_ubs()])
    UBS.validate(tabela)
    linha = tabela.row(0, named=True)
    assert linha["cnes"] == "0000302"
    assert linha["uf_codigo"] == "26"
    assert linha["latitude"] == pytest.approx(-8.21811)
    assert linha["coordenada_aproximada"] is False
    assert sum(descartes.values()) == 0


@pytest.mark.parametrize(
    ("campos", "motivo"),
    [
        ({"cnes": "nan"}, "CNES inválido"),
        ({"ibge": "12345"}, "código IBGE inválido"),
        ({"latitude": "nan"}, "sem coordenada"),
        ({"latitude": "40,1"}, "fora do Brasil"),
    ],
)
def test_descarta_por_motivo(campos, motivo):
    tabela, descartes = normalizar_ubs([_ubs(**campos)])
    assert tabela.height == 0
    assert descartes[motivo] == 1


def test_descarta_cnes_repetido_mantendo_o_primeiro():
    tabela, descartes = normalizar_ubs([_ubs(), _ubs(nome="OUTRA")])
    assert tabela["nome"].to_list() == ["USF SANTO ESTEVAO"]
    assert descartes["CNES repetido"] == 1


def test_marca_coordenada_aproximada():
    tabela, _ = normalizar_ubs([_ubs(latitude="-8,22", longitude=-35.2)])
    assert tabela["coordenada_aproximada"].to_list() == [True]


def test_tabela_vazia_mantem_o_esquema():
    tabela, _ = normalizar_ubs([])
    assert tabela.height == 0
    assert "coordenada_aproximada" in tabela.columns
