"""Ocupações (CBO) de atendimento exibidas no mapa, agrupadas por área.

Python puro: a aplicação também importa este módulo, e o Polars não vai para o deploy.
"""

# Grupos exibidos no mapa, na ordem de exibição, pelas famílias da CBO (4 primeiros
# dígitos). Ocupações administrativas e de apoio (recepção, limpeza, motorista) ficam de
# fora: o mapa mostra quem presta atendimento de saúde.
GRUPOS: dict[str, tuple[str, ...]] = {
    # 2231 é a família antiga dos médicos (inclui o médico residente, 2231F9); 2251 a
    # 2253 são clínicas, cirúrgicas e de diagnóstico e terapêutica.
    "Médicos": ("2231", "2251", "2252", "2253"),
    "Enfermagem": ("2235", "3222"),
    "Saúde bucal": ("2232", "3224"),
    # Farmácia, fisioterapia, nutrição, fonoaudiologia, terapia ocupacional, educação
    # física, osteopatia e quiropraxia, terapias integrativas, psicologia e serviço social.
    "Equipe multiprofissional": (
        "2234",
        "2236",
        "2237",
        "2238",
        "2239",
        "2241",
        "2261",
        "2263",
        "2515",
        "2516",
    ),
    # Agentes comunitários de saúde e de combate às endemias, entre outros.
    "Agentes de saúde": ("5151",),
}
_GRUPO_DA_FAMILIA = {familia: grupo for grupo, familias in GRUPOS.items() for familia in familias}

# Ocupações recentes que faltam na tabela CBO do DATASUS (TAB_CNES, 2026-11) ou que fogem
# do grupo da família. O 3222-55 está na família da enfermagem, mas os profissionais
# lançados com ele em PE (2026-09) têm o perfil dos agentes comunitários: 98% sem
# registro em conselho (os técnicos de enfermagem têm COREN) e lotados em equipes de
# saúde da família.
OCUPACOES_COMPLEMENTARES = {"322255": "Técnico em agente comunitário de saúde"}
_GRUPO_DO_CODIGO = {"322255": "Agentes de saúde"}


def grupo_da_ocupacao(cbo: str) -> str | None:
    """Grupo de exibição de um código CBO, ou None se não for ocupação de atendimento."""
    return _GRUPO_DO_CODIGO.get(cbo) or _GRUPO_DA_FAMILIA.get(cbo[:4])
