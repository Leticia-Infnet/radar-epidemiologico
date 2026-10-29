import httpx
import pandas as pd
import pytest

from app import geo
from app.geo import (
    CepInvalido,
    CepNaoEncontrado,
    Localizacao,
    construir_indice,
    detalhar_estabelecimento,
    geocodificar_cep,
    mapa_ubs,
    normalizar_cep,
    ubs_mais_proximas,
)
from common.config import Settings

UBS = pd.DataFrame(
    {
        "cnes": ["0000001", "0000002", "0000003"],
        "nome": ["UBS SAO PAULO", "UBS RIO", "UBS RECIFE"],
        "logradouro": ["RUA A", "RUA B", "RUA C"],
        "bairro": ["CENTRO", None, "BOA VISTA"],
        "latitude": [-23.5505, -22.9068, -8.0476],
        "longitude": [-46.6333, -43.1729, -34.8770],
        "coordenada_aproximada": [False, True, False],
    }
)

BRASILAPI_OK = {
    "street": "Rua Prudente de Morais",
    "neighborhood": "Carmo",
    "city": "Olinda",
    "state": "PE",
    "location": {"coordinates": {"latitude": "-8.00889", "longitude": "-34.85528"}},
}
AWESOME_OK = {
    "address": "Rua Prudente de Morais",
    "district": "Carmo",
    "city": "Olinda",
    "state": "PE",
    "lat": "-8.0155082",
    "lng": "-34.8519549",
}
VIACEP_OK = {
    "logradouro": "Rua Prudente de Morais",
    "bairro": "Carmo",
    "localidade": "Olinda",
    "uf": "PE",
}
NOMINATIM_OK = [{"lat": "-8.0150", "lon": "-34.8520"}]


def _cliente(respostas: dict[str, tuple[int, object]]) -> httpx.Client:
    """Cliente simulado: responde por host; host ausente gera erro de conexão."""

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.host not in respostas:
            raise httpx.ConnectError("sem rota", request=request)
        status, corpo = respostas[request.url.host]
        return httpx.Response(status, json=corpo)

    return httpx.Client(transport=httpx.MockTransport(responder))


@pytest.mark.parametrize("entrada", ["53020-140", "53020140", " 53020 140 "])
def test_normalizar_cep(entrada):
    assert normalizar_cep(entrada) == "53020140"


@pytest.mark.parametrize("entrada", ["", "5302014", "530201400", "abc"])
def test_cep_invalido(entrada):
    with pytest.raises(CepInvalido):
        normalizar_cep(entrada)


def test_geocodifica_pela_brasilapi():
    local = geocodificar_cep("53020-140", _cliente({"brasilapi.com.br": (200, BRASILAPI_OK)}))
    assert local.fonte == "BrasilAPI"
    assert local.latitude == pytest.approx(-8.00889)
    assert local.endereco == "Rua Prudente de Morais, Carmo, Olinda, PE"


def test_brasilapi_sem_coordenadas_cai_para_awesomeapi():
    sem_coords = {**BRASILAPI_OK, "location": {"coordinates": {}}}
    cliente = _cliente(
        {"brasilapi.com.br": (200, sem_coords), "cep.awesomeapi.com.br": (200, AWESOME_OK)}
    )
    assert geocodificar_cep("53020140", cliente).fonte == "AwesomeAPI"


def test_cai_para_viacep_e_nominatim():
    cliente = _cliente(
        {
            "brasilapi.com.br": (500, {}),
            "cep.awesomeapi.com.br": (404, {"code": "not_found"}),
            "viacep.com.br": (200, VIACEP_OK),
            "nominatim.openstreetmap.org": (200, NOMINATIM_OK),
        }
    )
    local = geocodificar_cep("53020140", cliente)
    assert local.fonte == "ViaCEP + OpenStreetMap"
    assert local.longitude == pytest.approx(-34.852)


def test_coordenada_fora_do_brasil_e_ignorada():
    fora = {**BRASILAPI_OK, "location": {"coordinates": {"latitude": "40.7", "longitude": "-74"}}}
    cliente = _cliente(
        {"brasilapi.com.br": (200, fora), "cep.awesomeapi.com.br": (200, AWESOME_OK)}
    )
    assert geocodificar_cep("53020140", cliente).fonte == "AwesomeAPI"


def test_nenhum_servico_encontra():
    cliente = _cliente({"viacep.com.br": (200, {"erro": "true"})})
    with pytest.raises(CepNaoEncontrado):
        geocodificar_cep("69900000", cliente)


def test_detalhar_estabelecimento():
    payload = {
        "codigo_cnes": 302,
        "nome_fantasia": "USF SANTO ESTEVAO",
        "endereco_estabelecimento": "RUA DO CEMITERIO",
        "numero_estabelecimento": "68",
        "bairro_estabelecimento": "PONTE DOS CARVALHOS",
        "numero_telefone_estabelecimento": "(81)35222744",
        "endereco_email_estabelecimento": None,
        "descricao_turno_atendimento": "ATENDIMENTOS NOS TURNOS DA MANHA E A TARDE",
    }
    cliente = httpx.Client(
        base_url="https://demas.teste",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    detalhe = detalhar_estabelecimento("0000302", cliente, Settings())
    assert detalhe["cnes"] == "0000302"
    assert detalhe["endereco"] == "RUA DO CEMITERIO, 68"
    assert detalhe["email"] is None


def test_detalhar_estabelecimento_inexistente():
    cliente = httpx.Client(
        base_url="https://demas.teste",
        transport=httpx.MockTransport(lambda r: httpx.Response(404, json={"message": "x"})),
    )
    assert detalhar_estabelecimento("9999999", cliente, Settings()) is None


def test_ubs_mais_proxima_e_distancia():
    indice = construir_indice(UBS)
    # Ponto em Campinas: São Paulo deve vir antes do Rio.
    resultado = ubs_mais_proximas(indice, UBS, -22.9056, -47.0608, k=2)
    assert list(resultado["cnes"]) == ["0000001", "0000002"]
    assert resultado["distancia_km"].iloc[0] == pytest.approx(84, abs=5)


def test_k_fora_do_intervalo():
    with pytest.raises(ValueError):
        ubs_mais_proximas(construir_indice(UBS), UBS, 0, 0, k=6)


def test_mapa_tem_origem_e_ubs_e_escapa_html():
    origem = Localizacao(-23.0, -46.0, "BrasilAPI", "Rua X")
    proximas = ubs_mais_proximas(construir_indice(UBS), UBS, -23.0, -46.0, k=2)
    detalhes = {"0000001": {"nome": "<script>alert(1)</script>"}, "0000002": None}
    html = mapa_ubs(origem, proximas, detalhes).get_root().render()
    assert html.count("L.marker(") == 3
    assert "<script>alert(1)</script>" not in html
    assert "Localização aproximada" in html


@pytest.mark.parametrize("valor", [None, float("nan"), pd.NA, "nan", "  "])
def test_texto_vazio(valor):
    assert geo.texto(valor) == ""


def test_link_rota():
    url = geo.link_rota(Localizacao(-8.0, -34.8, "x", ""), -8.01, -34.85)
    assert url.startswith("https://www.openstreetmap.org/directions?")
    assert "-8.00000%2C-34.80000%3B-8.01000%2C-34.85000" in url
