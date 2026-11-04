import polars as pl
import pytest

from etl import pipeline
from etl.extract.datasus import UFS, competencia_completa
from etl.transform.cbo import grupo_da_ocupacao
from etl.transform.profissionais import agregar_profissionais, competencia_iso
from etl.transform.schemas import ProfissionalUBS

OCUPACOES = {
    "225124": "Médico pediatra",
    "223565": "Enfermeiro da estratégia de saúde da família",
    "515105": "Agente comunitário de saúde",
    "411010": "Assistente administrativo",
}


def test_competencia_completa_ignora_mes_com_ufs_faltando():
    nomes = [f"/PF/PF{uf}2608.dbc" for uf in UFS] + [f"PF{uf}2609.DBC" for uf in UFS[:-1]]
    nomes += ["/PF/LEIAME.txt", "/PF/PFXX99.dbc"]
    assert competencia_completa(nomes) == "2608"
    with pytest.raises(LookupError):
        competencia_completa(nomes[-5:])


def test_competencia_iso():
    assert competencia_iso("2609") == "2026-09"


@pytest.mark.parametrize(
    ("cbo", "grupo"),
    [
        ("225124", "Médicos"),
        ("2231F9", "Médicos"),  # médico residente, família antiga da CBO
        ("322245", "Enfermagem"),
        ("223208", "Saúde bucal"),
        ("251510", "Equipe multiprofissional"),
        ("515140", "Agentes de saúde"),
        ("322255", "Agentes de saúde"),  # técnico em ACS, embora na família 3222
        ("411010", None),  # administrativo
        ("782305", None),  # motorista
    ],
)
def test_grupo_da_ocupacao(cbo, grupo):
    assert grupo_da_ocupacao(cbo) == grupo


def test_agrega_profissionais_distintos_sem_dados_pessoais():
    vinculos = [
        # Mesmo pediatra com dois vínculos na unidade conta uma vez.
        ("0000302", "225124", "700000000000001", "x1"),
        ("0000302", "225124", "700000000000001", "x1"),
        ("0000302", "225124", "700000000000002", "x2"),
        # Sem CNS, identifica pelo CPF codificado.
        ("302", "223565", "", "x3"),
        ("0000302", "515105", "", ""),  # sem identificação: ignorado
        ("0000302", "411010", "700000000000004", "x4"),  # administrativo: ignorado
        ("9999999", "225124", "700000000000005", "x5"),  # não é UBS: ignorado
        ("0000302", "322255", "700000000000006", "x6"),  # fora da tabela CBO
        ("0000302", "225299", "700000000000007", "x7"),  # código desconhecido
    ]
    tabela = agregar_profissionais(
        vinculos,
        {"0000302"},
        OCUPACOES,
        "2026-09",
        {"2252": "Médicos em especialidades cirúrgicas"},
    )
    ProfissionalUBS.validate(tabela)
    assert tabela.drop("competencia").rows() == [
        ("0000302", "Agentes de saúde", "322255", "Técnico em agente comunitário de saúde", 1),
        ("0000302", "Enfermagem", "223565", OCUPACOES["223565"], 1),
        ("0000302", "Médicos", "225124", "Médico pediatra", 2),
        ("0000302", "Médicos", "225299", "Médicos em especialidades cirúrgicas (CBO 225299)", 1),
    ]
    assert not {"CNS_PROF", "CPF_PROF", "NOMEPROF"} & set(tabela.columns)


def test_profissionais_abaixo_do_piso_nao_publica(monkeypatch):
    publicacoes = []
    monkeypatch.setattr(pipeline, "_cnes_das_ubs_publicadas", lambda: {"0000302"})
    monkeypatch.setattr(pipeline.datasus, "competencia_pf", lambda: "2609")
    monkeypatch.setattr(pipeline.datasus, "ocupacoes_cbo", lambda: (OCUPACOES, {}))
    monkeypatch.setattr(
        pipeline.datasus,
        "profissionais_uf",
        lambda uf, aamm, campos: iter([("0000302", "225124", "7001", "")]),
    )
    monkeypatch.setattr(pipeline, "publicar_pasta", lambda *a, **k: publicacoes.append(a))
    with pytest.raises(pipeline.ExtracaoIncompleta):
        pipeline.executar_profissionais(publicar=True)
    assert publicacoes == []


def test_esquema_recusa_contagem_zero():
    tabela = pl.DataFrame(
        {
            "cnes": ["0000302"],
            "grupo": ["Médicos"],
            "cbo": ["225124"],
            "ocupacao": ["Médico pediatra"],
            "profissionais": [0],
            "competencia": ["2026-09"],
        }
    )
    with pytest.raises(Exception, match="profissionais"):
        ProfissionalUBS.validate(tabela)
