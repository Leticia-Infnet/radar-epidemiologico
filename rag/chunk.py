"""Divisão dos manuais em trechos para a busca, com a seção e as páginas de cada um.

Entrada: os blocos de um manual em ordem de leitura (títulos, parágrafos e tabelas, com a
página do PDF), extraídos por `rag.ingest`. Saída: trechos de até `MAX_TOKENS` tokens do
tokenizer do e5, com `SOBREPOSICAO` tokens repetidos entre trechos seguidos da mesma seção.

O e5 trunca a entrada em 512 tokens; com o cabeçalho (manual e seção) e o prefixo
`passage: `, o conteúdo precisa ficar perto de 400. A seção 7.4 do AGENTS.md falava em 500
a 800 tokens, o que não caberia no modelo.

Puro: sem Docling nem modelo, para rodar nos testes e ser importado pelo app.
"""

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace

from rag.texto import normalizar

MAX_TOKENS = 400  # conteúdo; com cabeçalho e "passage: " fica abaixo de 512
SOBREPOSICAO = 60  # cerca de 15%
MIN_TOKENS = 20  # trechos menores são descartados
MAX_CARACTERES_TITULO = 200  # "título" mais longo que isto é parágrafo mal classificado
MAX_CARACTERES_SECAO = 150  # no cabeçalho indexado
SECAO_INICIAL = "Introdução"

# Seções que só atrapalham a busca: o sumário, por exemplo, tem o nome de todas as outras.
SECOES_DESCARTADAS = (
    "sumario",
    "ficha catalografica",
    "referencias",
    "bibliografia",
    "lista de",
    "equipe",
    "colaboradores",
    "elaboracao",
    "expediente",
)
MARCAS_DE_EXPEDIENTE = ("isbn", "creative commons", "dados internacionais de catalogacao")

_FIM_DE_FRASE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9•(–-])")
_SEPARADOR_TABELA = re.compile(r"^\|?\s*:?-{3,}")


@dataclass(frozen=True)
class Bloco:
    """Um item do PDF em ordem de leitura."""

    manual: str
    pagina: int  # página do PDF, a partir de 1
    tipo: str  # "titulo" | "texto" | "tabela"
    texto: str


@dataclass(frozen=True)
class Secao:
    manual: str
    titulo: str
    blocos: tuple[Bloco, ...]


@dataclass(frozen=True)
class Trecho:
    id: str
    manual: str
    pagina_inicial: int
    pagina_final: int
    secao: str
    texto: str
    n_tokens: int


@dataclass(frozen=True)
class _Unidade:
    """Frase, pedaço de tabela ou janela de palavras: o menor pedaço empacotado."""

    texto: str
    pagina: int
    tokens: int
    bloco: int  # posição do bloco na seção: unidades do mesmo bloco se juntam com espaço


def _caractere_privado(marcador: re.Match[str]) -> str:
    codigo = ord(marcador.group())
    # U+F700 a U+F7FF: fontes "expert" que deslocam o ASCII e o Latin-1 (\uf761 é "a").
    if 0xF720 <= codigo <= 0xF7FF:
        return chr(codigo - 0xF700)
    return " "  # setas e marcadores de fontes de símbolos


def limpar(texto: str) -> str:
    """Decodifica glifos de uso privado, desfaz a hifenização de fim de linha e colapsa
    quebras e espaços."""
    texto = re.sub(r"[\ue000-\uf8ff]", _caractere_privado, texto)
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)
    # "manifesta- ções": hífen solto seguido de continuação minúscula de 2+ letras.
    texto = re.sub(r"([a-zà-ú])- ([a-zà-ú]{2,})", r"\1\2", texto)
    return re.sub(r"\s+", " ", texto).strip()


def limpar_titulo(texto: str) -> str:
    """Título sem marcadores no início ("}AGENTES ETIOLÓGICOS" vira "AGENTES ETIOLÓGICOS")."""
    return re.sub(r"^[^\w(]+", "", limpar(texto))


def remover_repetidos(blocos: list[Bloco], fracao: float = 0.3) -> list[Bloco]:
    """Tira textos curtos repetidos em mais de `fracao` das páginas de um manual.

    São cabeçalhos e rodapés (título do manual, nome da seção corrente) que o Docling não
    marcou como tais. O número da página, no início ou no fim do texto, é ignorado na
    comparação: "20 | Ministério da Saúde" e "22 | Ministério da Saúde" são o mesmo cabeçalho.
    """

    def chave(texto: str) -> str:
        return re.sub(r"^\d+|\d+$", "#", normalizar(texto))

    paginas_do_manual: dict[str, set[int]] = defaultdict(set)
    paginas_do_texto: dict[tuple[str, str], set[int]] = defaultdict(set)
    for b in blocos:
        paginas_do_manual[b.manual].add(b.pagina)
        if len(b.texto) <= MAX_CARACTERES_TITULO:
            paginas_do_texto[b.manual, chave(b.texto)].add(b.pagina)

    def repetido(b: Bloco) -> bool:
        total = len(paginas_do_manual[b.manual])
        if total < 5 or len(b.texto) > MAX_CARACTERES_TITULO:
            return False
        return len(paginas_do_texto[b.manual, chave(b.texto)]) / total > fracao

    return [b for b in blocos if not repetido(b)]


def secionar(blocos: list[Bloco]) -> list[Secao]:
    """Agrupa os blocos pelo título mais recente.

    Títulos seguidos, sem conteúdo entre eles (capítulo e seção, por exemplo), se juntam com
    " › ", guardando os dois últimos. O conteúdo antes do primeiro título fica em
    "Introdução".
    """
    secoes: list[Secao] = []
    pendentes: list[str] = []
    titulo = SECAO_INICIAL
    atuais: list[Bloco] = []
    for b in blocos:
        if b.tipo == "titulo" and len(b.texto) <= MAX_CARACTERES_TITULO:
            if atuais:
                secoes.append(Secao(atuais[0].manual, titulo, tuple(atuais)))
                atuais, pendentes = [], []
            pendentes.append(limpar_titulo(b.texto))
            titulo = " › ".join(pendentes[-2:])
            continue
        if b.tipo == "titulo":
            b = replace(b, tipo="texto")
        atuais.append(b)
    if atuais:
        secoes.append(Secao(atuais[0].manual, titulo, tuple(atuais)))
    return secoes


def descartavel(secao: Secao) -> bool:
    """Sumário, ficha catalográfica, referências, expediente e linhas de sumário."""
    titulo = re.sub(r"^[\d.\s]+", "", normalizar(secao.titulo))
    if titulo.startswith(SECOES_DESCARTADAS) or "ficha" in titulo:
        return True
    if re.search(r"(\|\s*\d+|\.{4,}\s*\d+)$", titulo):  # "Apêndices | 69"
        return True
    texto = normalizar(" ".join(b.texto for b in secao.blocos))
    return any(marca in texto for marca in MARCAS_DE_EXPEDIENTE)


def filtrar_secoes(secoes: list[Secao]) -> list[Secao]:
    """Tira as seções que não servem de fonte para o chatbot."""
    return [s for s in secoes if not descartavel(s)]


def _janelas(
    texto: str, pagina: int, bloco: int, contar: Callable[[str], int], maximo: int
) -> list[_Unidade]:
    """Corta um texto longo em janelas de palavras que caibam em `maximo` tokens."""
    unidades: list[_Unidade] = []
    palavras: list[str] = []
    soma = 0
    for palavra in texto.split():
        n = contar(palavra)
        if palavras and soma + n > maximo:
            unidades.append(_Unidade(" ".join(palavras), pagina, soma, bloco))
            palavras, soma = [], 0
        palavras.append(palavra)
        soma += n
    if palavras:
        unidades.append(_Unidade(" ".join(palavras), pagina, soma, bloco))
    return unidades


def _unidades_de_texto(
    b: Bloco, indice: int, contar: Callable[[str], int], maximo: int
) -> list[_Unidade]:
    unidades: list[_Unidade] = []
    for frase in _FIM_DE_FRASE.split(b.texto):
        frase = frase.strip()
        if not frase:
            continue
        n = contar(frase)
        if n > maximo:
            unidades.extend(_janelas(frase, b.pagina, indice, contar, maximo))
        else:
            unidades.append(_Unidade(frase, b.pagina, n, indice))
    return unidades


def _unidades_de_tabela(
    b: Bloco, indice: int, contar: Callable[[str], int], maximo: int
) -> list[_Unidade]:
    """Pedaços da tabela em markdown, cada um com a linha de cabeçalho repetida."""
    linhas = [linha for linha in b.texto.splitlines() if linha.strip()]
    if contar(b.texto) <= maximo or not linhas:
        return [_Unidade(b.texto.strip(), b.pagina, contar(b.texto), indice)] if linhas else []
    tem_cabecalho = len(linhas) > 1 and _SEPARADOR_TABELA.match(linhas[1].strip())
    cabecalho = linhas[:2] if tem_cabecalho else []
    corpo = linhas[2:] if tem_cabecalho else linhas
    custo_cabecalho = contar("\n".join(cabecalho)) if cabecalho else 0

    unidades: list[_Unidade] = []
    pedaco: list[str] = []
    soma = custo_cabecalho
    for linha in corpo:
        n = contar(linha)
        if custo_cabecalho + n > maximo:  # uma linha sozinha não cabe: vira janelas
            if pedaco:
                unidades.append(_Unidade("\n".join(cabecalho + pedaco), b.pagina, soma, indice))
                pedaco, soma = [], custo_cabecalho
            unidades.extend(_janelas(linha, b.pagina, indice, contar, maximo))
            continue
        if pedaco and soma + n > maximo:
            unidades.append(_Unidade("\n".join(cabecalho + pedaco), b.pagina, soma, indice))
            pedaco, soma = [], custo_cabecalho
        pedaco.append(linha)
        soma += n
    if pedaco:
        unidades.append(_Unidade("\n".join(cabecalho + pedaco), b.pagina, soma, indice))
    # Cada pedaço da tabela fica separado dos demais por quebra de linha: identificadores
    # negativos não colidem com as posições dos outros blocos.
    return [replace(u, bloco=-(indice * 10_000 + i + 1)) for i, u in enumerate(unidades)]


def _juntar(unidades: list[_Unidade]) -> str:
    partes: list[str] = []
    for anterior, u in zip([None, *unidades[:-1]], unidades, strict=True):
        if anterior is not None:
            partes.append(" " if anterior.bloco == u.bloco else "\n")
        partes.append(u.texto)
    return "".join(partes)


def dividir(
    secoes: list[Secao],
    contar_tokens: Callable[[str], int],
    max_tokens: int = MAX_TOKENS,
    sobreposicao: int = SOBREPOSICAO,
) -> list[Trecho]:
    """Empacota as unidades de cada seção em trechos de até `max_tokens`.

    Um trecho novo da mesma seção começa com as últimas unidades do anterior que somem até
    `sobreposicao` tokens. Trechos nunca cruzam seções.
    """
    trechos: list[Trecho] = []
    contagem: dict[str, int] = defaultdict(int)

    def emitir(secao: Secao, unidades: list[_Unidade]) -> None:
        texto = _juntar(unidades)
        n = contar_tokens(texto)
        if n < MIN_TOKENS:
            return
        contagem[secao.manual] += 1
        trechos.append(
            Trecho(
                id=f"{secao.manual}-{contagem[secao.manual]:04d}",
                manual=secao.manual,
                pagina_inicial=min(u.pagina for u in unidades),
                pagina_final=max(u.pagina for u in unidades),
                secao=secao.titulo,
                texto=texto,
                n_tokens=n,
            )
        )

    for secao in secoes:
        unidades: list[_Unidade] = []
        for i, b in enumerate(secao.blocos):
            if b.tipo == "tabela":
                unidades.extend(_unidades_de_tabela(b, i, contar_tokens, max_tokens))
            else:
                unidades.extend(_unidades_de_texto(b, i, contar_tokens, max_tokens))

        atual: list[_Unidade] = []
        soma = 0
        for u in unidades:
            if atual and soma + u.tokens > max_tokens:
                emitir(secao, atual)
                sobra: list[_Unidade] = []
                soma_sobra = 0
                for v in reversed(atual[1:]):  # nunca o trecho anterior inteiro
                    if soma_sobra + v.tokens > sobreposicao:
                        break
                    sobra.insert(0, v)
                    soma_sobra += v.tokens
                while sobra and soma_sobra + u.tokens > max_tokens:
                    soma_sobra -= sobra.pop(0).tokens
                atual, soma = sobra, soma_sobra
            atual.append(u)
            soma += u.tokens
        if atual:
            emitir(secao, atual)
    return trechos


def texto_indexado(titulo_manual: str, secao: str, texto: str) -> str:
    """Texto que vai para o embedding e para o BM25: cabeçalho com manual e seção + trecho."""
    if len(secao) > MAX_CARACTERES_SECAO:
        secao = secao[: MAX_CARACTERES_SECAO - 1].rstrip() + "…"
    return f"{titulo_manual} — {secao}\n{texto}"
