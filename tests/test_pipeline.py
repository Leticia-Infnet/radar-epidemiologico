import pytest

from etl import pipeline


def test_extracao_abaixo_do_piso_nao_publica(monkeypatch):
    async def poucos_registros():
        return [{"cnes": "0000302"}]

    publicacoes = []
    monkeypatch.setattr(pipeline, "_baixar_ubs", poucos_registros)
    monkeypatch.setattr(pipeline, "publicar_pasta", lambda *a, **k: publicacoes.append(a))
    with pytest.raises(pipeline.ExtracaoIncompleta):
        pipeline.executar_ubs(publicar=True)
    assert publicacoes == []
