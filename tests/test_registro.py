import json
import threading
from types import SimpleNamespace

from common.config import Settings
from rag.registro import Registro, RegistroNulo, criar_registro, mascarar


def test_mascarar_dados_pessoais():
    texto = (
        "Sou Ana, CPF 123.456.789-09, e-mail ana.silva@exemplo.com.br, "
        "tel (81) 98765-4321, CEP 50050-000. Tenho febre há 3 dias."
    )
    mascarado = mascarar(texto)
    for dado in ("123.456.789-09", "ana.silva@", "98765-4321", "50050-000"):
        assert dado not in mascarado
    assert "[CPF]" in mascarado and "[EMAIL]" in mascarado
    assert "[TELEFONE]" in mascarado and "[CEP]" in mascarado
    assert "febre há 3 dias" in mascarado


def test_registro_grava_jsonl(tmp_path):
    registro = Registro(SimpleNamespace(lock=threading.Lock()), tmp_path)
    registro.registrar("Meu CPF é 12345678909, isso é dengue?", "abaixo_do_limiar", 0.71)
    registro.registrar("Outra pergunta", "sem_citacao", None)
    linhas = [json.loads(linha) for linha in registro.arquivo.read_text().splitlines()]
    assert [linha["motivo"] for linha in linhas] == ["abaixo_do_limiar", "sem_citacao"]
    assert linhas[0]["pergunta"] == "Meu CPF é [CPF], isso é dengue?"
    assert linhas[0]["similaridade"] == 0.71
    assert set(linhas[0]) == {"quando", "pergunta", "motivo", "similaridade"}


def test_sem_token_o_registro_e_nulo():
    assert isinstance(criar_registro(Settings(hf_consultas_token=None)), RegistroNulo)
