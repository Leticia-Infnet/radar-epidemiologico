from rag.chunk import (
    MIN_TOKENS,
    Bloco,
    Secao,
    dividir,
    filtrar_secoes,
    limpar,
    limpar_titulo,
    remover_repetidos,
    secionar,
    texto_indexado,
)


def contar(texto: str) -> int:
    return len(texto.split())


def frases(n: int, palavras: int = 10, prefixo: str = "Frase") -> str:
    """n frases de `palavras` palavras cada, numeradas."""
    return " ".join(
        f"{prefixo} {i} " + " ".join(["palavra"] * (palavras - 3)) + " fim." for i in range(n)
    )


def test_limpar_desfaz_hifenizacao_e_quebras():
    assert limpar("manifesta-\nções  clínicas\nda dengue") == "manifestações clínicas da dengue"
    assert limpar("manifesta- ções") == "manifestações"
    assert limpar("pré- e pós-natal") == "pré- e pós-natal"


def test_remover_repetidos_tira_cabecalho_de_pagina():
    blocos = [Bloco("m", p, "texto", "Guia de Vigilância em Saúde") for p in range(1, 11)]
    blocos += [Bloco("m", p, "texto", f"Conteúdo da página {p}.") for p in range(1, 11)]
    restantes = remover_repetidos(blocos)
    assert len(restantes) == 10
    assert all("Conteúdo" in b.texto for b in restantes)


def test_secionar_junta_titulos_seguidos_e_cria_introducao():
    blocos = [
        Bloco("m", 1, "texto", "Apresentação do manual."),
        Bloco("m", 2, "titulo", "Capítulo 7"),
        Bloco("m", 2, "titulo", "Dengue"),
        Bloco("m", 2, "texto", "A dengue é uma doença."),
        Bloco("m", 3, "titulo", "Sinais de alarme"),
        Bloco("m", 3, "texto", "Dor abdominal intensa."),
        Bloco("m", 4, "titulo", "x" * 250),  # longo demais: vira texto
    ]
    secoes = secionar(blocos)
    assert [s.titulo for s in secoes] == ["Introdução", "Capítulo 7 › Dengue", "Sinais de alarme"]
    assert secoes[2].blocos[-1].tipo == "texto"


def test_dividir_respeita_maximo_e_sobrepoe_na_mesma_secao():
    secao = Secao("m", "Seção A", (Bloco("m", 1, "texto", frases(30)),))
    trechos = dividir([secao], contar, max_tokens=50, sobreposicao=10)
    assert len(trechos) > 2
    assert all(t.n_tokens <= 50 for t in trechos)
    for anterior, seguinte in zip(trechos, trechos[1:], strict=False):
        ultima_frase = anterior.texto.split(". ")[-1]
        assert seguinte.texto.startswith(ultima_frase.rstrip("."))


def test_dividir_nao_sobrepoe_entre_secoes():
    a = Secao("m", "A", (Bloco("m", 1, "texto", frases(4, prefixo="Alfa")),))
    b = Secao("m", "B", (Bloco("m", 2, "texto", frases(4, prefixo="Beta")),))
    trechos = dividir([a, b], contar, max_tokens=100, sobreposicao=20)
    assert [t.secao for t in trechos] == ["A", "B"]
    assert "Alfa" not in trechos[1].texto


def test_dividir_paginas_e_ids():
    secao = Secao(
        "dengue",
        "S",
        (
            Bloco("dengue", 3, "texto", frases(3)),
            Bloco("dengue", 4, "texto", frases(3)),
        ),
    )
    (trecho,) = dividir([secao], contar, max_tokens=200)
    assert (trecho.pagina_inicial, trecho.pagina_final) == (3, 4)
    assert trecho.id == "dengue-0001"
    assert "\n" in trecho.texto  # blocos diferentes separados por quebra de linha


def test_dividir_tabela_repete_cabecalho():
    linhas = ["| Grupo | Conduta |", "|---|---|"]
    linhas += [f"| G{i} | " + " ".join(["hidratar"] * 8) + " |" for i in range(20)]
    secao = Secao("m", "Tabela", (Bloco("m", 5, "tabela", "\n".join(linhas)),))
    trechos = dividir([secao], contar, max_tokens=60, sobreposicao=0)
    assert len(trechos) > 1
    assert all(t.texto.startswith("| Grupo | Conduta |\n|---|---|") for t in trechos)


def test_dividir_descarta_trecho_curto():
    secao = Secao("m", "S", (Bloco("m", 1, "texto", "Curto demais."),))
    assert dividir([secao], contar) == []
    assert MIN_TOKENS > 2


def test_frase_longa_e_cortada_em_janelas():
    secao = Secao("m", "S", (Bloco("m", 1, "texto", " ".join(["palavra"] * 130)),))
    trechos = dividir([secao], contar, max_tokens=50, sobreposicao=0)
    assert [t.n_tokens for t in trechos] == [50, 50, 30]


def test_texto_indexado_corta_secao_longa():
    texto = texto_indexado("Manual", "s" * 300, "Conteúdo")
    cabecalho, conteudo = texto.split("\n")
    assert cabecalho.startswith("Manual — ") and len(cabecalho) < 170
    assert conteudo == "Conteúdo"


def test_limpar_decodifica_fonte_expert_e_tira_simbolos():
    apresentacao = "".join(chr(0xF700 + ord(c)) for c in "apresentação")
    assert limpar(apresentacao) == "apresentação"
    assert limpar("Dipirona - adultos 20 gotas") == "Dipirona - adultos 20 gotas"


def test_limpar_titulo_tira_marcador():
    assert limpar_titulo("}AGENTES ETIOLÓGICOS") == "AGENTES ETIOLÓGICOS"
    assert limpar_titulo("6.1 GRUPO A") == "6.1 GRUPO A"


def test_filtrar_secoes_tira_sumario_referencias_e_expediente():
    def secao(titulo, texto="Conteúdo útil."):
        return Secao("m", titulo, (Bloco("m", 1, "texto", texto),))

    secoes = [
        secao("Sumário › Apresentação | 7"),
        secao("6 Estadiamento clínico e conduta | 28"),
        secao("Ficha Catalográfica"),
        secao("Referências"),
        secao(
            "2015 Ministério da Saúde.",
            "Esta obra é disponibilizada nos termos da Creative Commons.",
        ),
        secao("Anexo B - Ficha Sinan - Verso"),
        secao("2.2.1 Dengue com sinais de alarme"),
    ]
    assert [s.titulo for s in filtrar_secoes(secoes)] == ["2.2.1 Dengue com sinais de alarme"]


def test_remover_repetidos_ignora_o_numero_da_pagina():
    blocos = [Bloco("m", p, "titulo", f"{p} | Ministério da Saúde") for p in range(2, 22, 2)]
    blocos += [Bloco("m", p, "texto", f"Conteúdo {p}.") for p in range(1, 21)]
    restantes = remover_repetidos(blocos)
    assert all("Ministério" not in b.texto for b in restantes)
    assert len(restantes) == 20
