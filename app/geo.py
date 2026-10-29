"""Geocodificação de CEP, busca das UBS mais próximas, detalhe do CNES e mapa Folium."""

import html
import logging
import re
from dataclasses import dataclass
from typing import Any

import folium
import httpx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from common.config import Settings, get_settings
from etl.transform.normalize import ausente, codigo_cnes, coordenada, dentro_do_brasil

log = logging.getLogger(__name__)

RAIO_TERRA_KM = 6371.0088
TIMEOUT_S = 15.0


class CepInvalido(ValueError):
    """O texto informado não é um CEP de 8 dígitos."""


class CepNaoEncontrado(LookupError):
    """Nenhum serviço devolveu coordenadas válidas para o CEP."""


@dataclass(frozen=True)
class Localizacao:
    """Ponto de partida da busca, com a origem da coordenada."""

    latitude: float
    longitude: float
    fonte: str
    endereco: str


def texto(valor: Any) -> str:
    """Texto limpo para exibição; vazio para None, NaN, pd.NA e a string "nan".

    Colunas opcionais do Parquet (bairro, logradouro) chegam ao pandas como NaN.
    """
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return ""
    return "" if ausente(valor) else str(valor).strip()


def normalizar_cep(cep: str) -> str:
    """Aceita `53020-140`, `53020140` ou com espaços; devolve os 8 dígitos."""
    digitos = re.sub(r"\D", "", cep or "")
    if len(digitos) != 8:
        raise CepInvalido(f"CEP deve ter 8 dígitos: {cep!r}")
    return digitos


def _endereco(*partes: Any) -> str:
    return ", ".join(str(p).strip() for p in partes if not ausente(p) and str(p).strip())


def _ponto(lat: Any, lon: Any, fonte: str, endereco: str) -> Localizacao | None:
    lat, lon = coordenada(lat), coordenada(lon)
    # A BrasilAPI já devolveu endereço inventado para CEP inexistente; a caixa do Brasil
    # não pega tudo, mas descarta coordenadas vazias, zeradas ou trocadas.
    if not dentro_do_brasil(lat, lon):
        return None
    return Localizacao(lat, lon, fonte, endereco)


def _brasilapi(cliente: httpx.Client, cep: str, s: Settings) -> Localizacao | None:
    r = cliente.get(s.brasilapi_cep_url.format(cep=cep))
    if r.status_code != 200:
        return None
    d = r.json()
    coords = (d.get("location") or {}).get("coordinates") or {}
    endereco = _endereco(d.get("street"), d.get("neighborhood"), d.get("city"), d.get("state"))
    return _ponto(coords.get("latitude"), coords.get("longitude"), "BrasilAPI", endereco)


def _awesomeapi(cliente: httpx.Client, cep: str, s: Settings) -> Localizacao | None:
    r = cliente.get(s.awesomeapi_cep_url.format(cep=cep))
    if r.status_code != 200:
        return None
    d = r.json()
    endereco = _endereco(d.get("address"), d.get("district"), d.get("city"), d.get("state"))
    return _ponto(d.get("lat"), d.get("lng"), "AwesomeAPI", endereco)


def _viacep_nominatim(cliente: httpx.Client, cep: str, s: Settings) -> Localizacao | None:
    r = cliente.get(s.viacep_url.format(cep=cep))
    if r.status_code != 200 or r.json().get("erro"):
        return None
    d = r.json()
    consulta = {"city": d.get("localidade"), "state": d.get("uf"), "country": "Brasil"}
    if d.get("logradouro"):
        consulta["street"] = d["logradouro"]
    # Política do Nominatim: no máximo 1 req/s e User-Agent identificável. É uma chamada
    # por CEP, e o resultado fica em cache na aplicação.
    r = cliente.get(s.nominatim_url, params={**consulta, "format": "jsonv2", "limit": 1})
    if r.status_code != 200 or not r.json():
        return None
    achado = r.json()[0]
    endereco = _endereco(d.get("logradouro"), d.get("bairro"), d.get("localidade"), d.get("uf"))
    return _ponto(achado.get("lat"), achado.get("lon"), "ViaCEP + OpenStreetMap", endereco)


PROVEDORES = (_brasilapi, _awesomeapi, _viacep_nominatim)


def geocodificar_cep(
    cep: str, cliente: httpx.Client | None = None, settings: Settings | None = None
) -> Localizacao:
    """Converte CEP em lat/lon: BrasilAPI, depois AwesomeAPI, depois ViaCEP + Nominatim.

    Levanta `CepInvalido` para formato errado e `CepNaoEncontrado` quando nenhum serviço
    devolve coordenada dentro do Brasil.
    """
    cep = normalizar_cep(cep)
    s = settings or get_settings()
    cliente = cliente or httpx.Client(
        timeout=TIMEOUT_S, headers={"User-Agent": s.user_agent}, follow_redirects=True
    )
    with cliente:
        for provedor in PROVEDORES:
            try:
                local = provedor(cliente, cep, s)
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("%s falhou para o CEP %s: %s", provedor.__name__, cep, exc)
                continue
            if local is not None:
                return local
    raise CepNaoEncontrado(cep)


def construir_indice(ubs: pd.DataFrame) -> BallTree:
    """Índice haversine sobre as coordenadas das UBS, em radianos."""
    return BallTree(np.radians(ubs[["latitude", "longitude"]].to_numpy()), metric="haversine")


def ubs_mais_proximas(
    indice: BallTree, ubs: pd.DataFrame, lat: float, lon: float, k: int = 5
) -> pd.DataFrame:
    """Devolve as k UBS mais próximas do ponto, com a distância em linha reta (km)."""
    if not 1 <= k <= 5:
        raise ValueError("k deve estar entre 1 e 5.")
    distancias, posicoes = indice.query(np.radians([[lat, lon]]), k=k)
    resultado = ubs.iloc[posicoes[0]].copy()
    resultado["distancia_km"] = distancias[0] * RAIO_TERRA_KM
    return resultado.reset_index(drop=True)


def detalhar_estabelecimento(
    cnes: str, cliente: httpx.Client | None = None, settings: Settings | None = None
) -> dict[str, Any] | None:
    """Consulta `/cnes/estabelecimentos/{cnes}` ao vivo; devolve None se indisponível."""
    s = settings or get_settings()
    cliente = cliente or httpx.Client(
        base_url=s.demas_base_url, timeout=TIMEOUT_S, headers={"User-Agent": s.user_agent}
    )
    try:
        with cliente:
            r = cliente.get(f"/cnes/estabelecimentos/{cnes}")
        if r.status_code != 200:
            return None
        d = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("CNES %s indisponível: %s", cnes, exc)
        return None

    def texto(campo: str) -> str | None:
        return None if ausente(d.get(campo)) else str(d[campo]).strip()

    return {
        "cnes": codigo_cnes(d.get("codigo_cnes")),
        "nome": texto("nome_fantasia"),
        "endereco": _endereco(texto("endereco_estabelecimento"), texto("numero_estabelecimento")),
        "bairro": texto("bairro_estabelecimento"),
        "cep": texto("codigo_cep_estabelecimento"),
        "telefone": texto("numero_telefone_estabelecimento"),
        "email": texto("endereco_email_estabelecimento"),
        "turno": texto("descricao_turno_atendimento"),
        "atualizado_em": texto("data_atualizacao"),
    }


def link_rota(origem: Localizacao, lat: float, lon: float) -> str:
    """Rota a pé no OpenStreetMap, da origem até a unidade."""
    return (
        "https://www.openstreetmap.org/directions?engine=fossgis_osrm_foot"
        f"&route={origem.latitude:.5f}%2C{origem.longitude:.5f}%3B{lat:.5f}%2C{lon:.5f}"
    )


def _popup(posicao: int, ubs: dict[str, Any], detalhe: dict[str, Any] | None) -> str:
    detalhe = detalhe or {}
    linhas = [
        f"<b>{posicao}. {html.escape(texto(detalhe.get('nome')) or texto(ubs['nome']))}</b>",
        f"{ubs['distancia_km']:.1f} km em linha reta",
        html.escape(texto(detalhe.get("endereco")) or texto(ubs.get("logradouro"))),
        html.escape(texto(detalhe.get("bairro")) or texto(ubs.get("bairro"))),
    ]
    if detalhe.get("telefone"):
        linhas.append(f"Telefone: {html.escape(detalhe['telefone'])}")
    if detalhe.get("turno"):
        linhas.append(html.escape(detalhe["turno"].capitalize()))
    if ubs.get("coordenada_aproximada"):
        linhas.append("<i>Localização aproximada no mapa.</i>")
    linhas.append(f"CNES {html.escape(ubs['cnes'])}")
    return "<br>".join(linha for linha in linhas if linha)


def mapa_ubs(
    origem: Localizacao,
    proximas: pd.DataFrame,
    detalhes: dict[str, dict[str, Any] | None],
) -> folium.Map:
    """Mapa OpenStreetMap com a origem e as UBS numeradas por proximidade."""
    mapa = folium.Map(location=[origem.latitude, origem.longitude], tiles="OpenStreetMap")
    folium.Marker(
        [origem.latitude, origem.longitude],
        tooltip="Seu CEP",
        popup=folium.Popup(html.escape(origem.endereco or "Seu CEP"), max_width=250),
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(mapa)
    for posicao, ubs in enumerate(proximas.to_dict("records"), start=1):
        folium.Marker(
            [ubs["latitude"], ubs["longitude"]],
            tooltip=f"{posicao}. {ubs['nome']} ({ubs['distancia_km']:.1f} km)",
            popup=folium.Popup(_popup(posicao, ubs, detalhes.get(ubs["cnes"])), max_width=300),
            icon=folium.Icon(
                color="orange" if ubs["coordenada_aproximada"] else "blue",
                icon="plus-square",
                prefix="fa",
            ),
        ).add_to(mapa)
    pontos = [[origem.latitude, origem.longitude]] + proximas[
        ["latitude", "longitude"]
    ].values.tolist()
    mapa.fit_bounds(pontos, padding=(30, 30))
    return mapa
