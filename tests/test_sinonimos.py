from rag.sinonimos import Sinonimo, carregar_sinonimos, expandir
from tests.test_retrieve import buscador_sintetico


def test_dicionario_versionado_compila():
    sinonimos = carregar_sinonimos()
    assert len(sinonimos) >= 20
    assert all(s.termos for s in sinonimos)


def test_expandir_acrescenta_termos_tecnicos():
    sinonimos = carregar_sinonimos()
    assert expandir("Grávida pode ter dengue?", sinonimos).endswith("gestante gestacao")
    assert "prurido" in expandir("Estou com COCEIRA no corpo", sinonimos)
    assert expandir("Qual a vacina?", sinonimos) == "Qual a vacina?"


def test_expandir_nao_repete_termos():
    import re

    regras = [Sinonimo(re.compile("a"), "x"), Sinonimo(re.compile("b"), "x")]
    assert expandir("a b", regras) == "a b x"


def test_sinonimos_ampliam_a_busca_sem_mudar_o_limiar():
    import re

    base = buscador_sintetico()
    regras = [Sinonimo(re.compile(r"\bgravid\w*"), "vacina dose unica febre amarela")]
    com = buscador_sintetico()
    com.sinonimos = regras
    pergunta = "grávida"
    assert not base.buscar(pergunta).resultados or base.buscar(pergunta).resultados[0].posicao != 2
    assert com.buscar(pergunta).resultados[0].posicao == 2  # o trecho da vacina
    # O limiar continua medido sobre a pergunta original.
    assert com.buscar(pergunta).similaridade_maxima == base.buscar(pergunta).similaridade_maxima
