"""Treino dos modelos de tendência, executado apenas no CI (workflow `train.yml`).

Para dengue e chikungunya: lê as séries semanais publicadas pelo ETL, compara os
candidatos de `ml.modelos` num backtest de origem móvel, escolhe o melhor, reajusta com
toda a série e publica no Hub (`previsao/`) as estimativas das próximas semanas com
intervalos, as métricas do backtest e o modelo. A aplicação só lê esses arquivos.

Uso: `python -m ml.train [--sem-publicar] [--doencas dengue chikungunya]`.
"""

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from huggingface_hub import HfFileSystem

from common.config import get_settings
from ml import dados
from ml.dados import SeriesSemanais, nivel_do_local
from ml.evaluate import (
    QUANTIS,
    Intervalos,
    backtest_origem_movel,
    dentro_do_intervalo,
    escalas_sazonais,
    metricas_por_local,
)
from ml.features import ATRIBUTOS, montar_atributos
from ml.modelos import BASELINES, CANDIDATOS, MODELOS

log = logging.getLogger(__name__)

# Além das semanas incompletas do fim da série, a estimativa avança este tanto.
SEMANAS_A_FRENTE = 4
# O primeiro ano de teste do backtest vem depois de dois anos de história para os
# atributos e de cerca de três anos de treino.
ANOS_ATE_O_PRIMEIRO_TESTE = 5
# Brasil e UFs são séries suaves; as municipais, ruidosas. Um modelo só para todas fica
# dominado pelos municípios e perde da persistência no Brasil, então cada grupo tem o seu.
GRUPO_DO_NIVEL = {"br": "agregado", "uf": "agregado", "municipio": "municipio"}
# Um modelo mais complexo só substitui o mais simples se o MASE mediano cair mais que isto.
MARGEM_DE_EMPATE = 0.02
# Piso de sanidade: BR e UFs mais os municípios elegíveis (1.260 e 120 em 2026-11).
MINIMO_DE_SERIES = {"dengue": 600, "chikungunya": 60}

COLUNAS_ESTIMATIVAS = [
    "doenca",
    "nivel",
    "local",
    "ano_epi",
    "semana_epi",
    "inicio_semana",
    "passo",
    "incompleta",
    "estimativa",
    *QUANTIS,
]
COLUNAS_METRICAS = [
    "doenca",
    "nivel",
    "local",
    "passo",
    "modelo",
    "mae",
    "mase",
    "cobertura_90",
    "n",
]


class ModeloReprovado(RuntimeError):
    """O treino não passou no portão de qualidade e nada deve ser publicado."""


@dataclass
class Treino:
    """Resultado do treino de uma doença."""

    doenca: str
    estimativas: pd.DataFrame
    metricas: pd.DataFrame
    modelos: dict[str, Any]
    resumo: dict[str, Any]


def escolher_modelo(mase_mediano: dict[str, float]) -> str:
    """O candidato de menor MASE mediano; na dúvida, o mais simples.

    `MODELOS` vai do mais simples ao mais complexo: o seguinte só toma o lugar se
    melhorar mais que `MARGEM_DE_EMPATE`.
    """
    escolhido, *demais = MODELOS
    for nome in demais:
        if mase_mediano[nome] < mase_mediano[escolhido] * (1 - MARGEM_DE_EMPATE):
            escolhido = nome
    return escolhido


def conferir_qualidade(treino: Treino, passos: int) -> None:
    """Portão de qualidade: levanta `ModeloReprovado` em vez de publicar algo ruim."""
    doenca, estimativas = treino.doenca, treino.estimativas
    series = estimativas["local"].nunique()
    if series < MINIMO_DE_SERIES.get(doenca, 0):
        raise ModeloReprovado(
            f"{doenca}: {series} séries, abaixo do mínimo de {MINIMO_DE_SERIES[doenca]}."
        )
    if len(estimativas) != series * passos:
        raise ModeloReprovado(f"{doenca}: faltam passos em alguma série.")
    numeros = estimativas[["estimativa", *QUANTIS]]
    if numeros.isna().any().any() or (numeros < 0).any().any():
        raise ModeloReprovado(f"{doenca}: há estimativa vazia ou negativa.")
    for grupo, info in treino.resumo["grupos"].items():
        mase = info["mase_mediano"]
        for baseline in BASELINES:
            if not mase[info["modelo"]] < mase[baseline]:
                raise ModeloReprovado(
                    f"{doenca}/{grupo}: {info['modelo']} (MASE {mase[info['modelo']]:.3f}) não "
                    f"supera {baseline} ({mase[baseline]:.3f})."
                )


def _arredondar(valor: float) -> float | None:
    """Número com 4 casas para o `metadados.json`; vazio vira `null`."""
    return None if pd.isna(valor) else round(float(valor), 4)


def treinar(series: SeriesSemanais) -> Treino:
    """Backtest, escolha do modelo e estimativas das próximas semanas de uma doença."""
    doenca, completas = series.doenca, series.completas
    n = series.semanas_completas
    incompletas = len(series.calendario) - n
    passos = range(1, incompletas + SEMANAS_A_FRENTE + 1)
    calendario = dados.estender_calendario(series.calendario, SEMANAS_A_FRENTE)
    primeiro_ano_teste = int(calendario["ano_epi"].min()) + ANOS_ATE_O_PRIMEIRO_TESTE
    niveis = {local: nivel_do_local(local) for local in completas.columns}
    grupos = {local: GRUPO_DO_NIVEL[nivel] for local, nivel in niveis.items()}
    escalas = escalas_sazonais(completas)

    metricas, estimativas = [], []
    ajustados: dict[tuple[str, str, int], dict[str, Any]] = {}
    for passo in passos:
        tabela = montar_atributos(completas, calendario, passo)
        grupo_da_linha = tabela["local"].astype(str).map(grupos)
        for grupo in sorted(set(grupos.values())):
            do_grupo = tabela[(grupo_da_linha == grupo).to_numpy()]
            previsoes = backtest_origem_movel(do_grupo, CANDIDATOS, primeiro_ano_teste)
            do_passo = metricas_por_local(previsoes, escalas, list(CANDIDATOS))
            do_passo["passo"] = passo
            do_passo["cobertura_90"] = float("nan")

            conhecidas = do_grupo[do_grupo["y"].notna()]
            futuro = do_grupo[do_grupo["origem"] == n - 1]
            for nome, criar in MODELOS.items():
                cobertura = (
                    dentro_do_intervalo(previsoes, nome)
                    .groupby(previsoes["local"].astype(str))
                    .mean()
                )
                do_modelo = do_passo["modelo"] == nome
                do_passo.loc[do_modelo, "cobertura_90"] = (
                    do_passo.loc[do_modelo, "local"].map(cobertura).to_numpy()
                )

                modelo = criar().ajustar(conhecidas)
                intervalos = Intervalos.ajustar(
                    previsoes["y"].to_numpy(),
                    previsoes[nome].to_numpy(),
                    previsoes["nivel_52"].to_numpy(),
                )
                estimativa = modelo.prever(futuro)
                limites = intervalos.aplicar(estimativa, futuro["nivel_52"].to_numpy())
                estimativas.append(
                    pd.DataFrame(
                        {
                            "modelo": nome,
                            "grupo": grupo,
                            "local": futuro["local"].astype(str).to_numpy(),
                            "passo": passo,
                            "alvo_pos": futuro["alvo_pos"].to_numpy(),
                            "estimativa": estimativa,
                        }
                    ).join(limites)
                )
                ajustados[(grupo, nome, passo)] = {"modelo": modelo, "intervalos": intervalos}
            do_passo["grupo"] = grupo
            metricas.append(do_passo)
        log.info("%s: passo %d de %d avaliado.", doenca, passo, len(passos))

    metricas = pd.concat(metricas, ignore_index=True)
    estimativas = pd.concat(estimativas, ignore_index=True)

    resumo_grupos, escolhidos = {}, {}
    for grupo, do_grupo in metricas.groupby("grupo"):
        mase_mediano = do_grupo.groupby("modelo")["mase"].median()
        escolhido = escolher_modelo(mase_mediano.to_dict())
        escolhidos[grupo] = escolhido
        do_escolhido = do_grupo[do_grupo["modelo"] == escolhido]
        resumo_grupos[grupo] = {
            "modelo": escolhido,
            "series": int(do_grupo["local"].nunique()),
            "mase_mediano": {m: _arredondar(v) for m, v in mase_mediano.items()},
            "mase_mediano_por_passo": {
                int(passo): _arredondar(v)
                for passo, v in do_escolhido.groupby("passo")["mase"].median().items()
            },
            # Média entre os locais; vazia se o backtest teve um único ano de teste.
            "cobertura_90": _arredondar(do_escolhido["cobertura_90"].mean()),
        }
        log.info("%s/%s: %s", doenca, grupo, json.dumps(resumo_grupos[grupo]))

    estimativas = estimativas[estimativas["modelo"] == estimativas["grupo"].map(escolhidos)]
    alvo = calendario.iloc[estimativas["alvo_pos"].to_numpy()]
    estimativas = estimativas.assign(
        doenca=doenca,
        nivel=estimativas["local"].map(niveis),
        ano_epi=alvo["ano_epi"].to_numpy(),
        semana_epi=alvo["semana_epi"].to_numpy(),
        inicio_semana=alvo["inicio_semana"].dt.date.to_numpy(),
        incompleta=estimativas["alvo_pos"].to_numpy() < len(series.calendario),
    )
    estimativas = estimativas.sort_values(["nivel", "local", "passo"], ignore_index=True)
    metricas = metricas.assign(doenca=doenca, nivel=metricas["local"].map(niveis))

    ultima = series.calendario.iloc[n - 1]
    resumo = {
        "ultima_semana_completa": f"{ultima['ano_epi']}-{ultima['semana_epi']:02d}",
        "inicio_da_ultima_semana_completa": ultima["inicio_semana"].date().isoformat(),
        "semanas_incompletas": incompletas,
        "passos": len(passos),
        "primeiro_ano_teste": primeiro_ano_teste,
        "series": pd.Series(niveis).value_counts().to_dict(),
        "grupos": resumo_grupos,
    }
    modelos = {
        "doenca": doenca,
        "atributos": ATRIBUTOS,
        "grupos": {
            grupo: {
                "modelo": nome,
                "passos": {passo: ajustados[(grupo, nome, passo)] for passo in passos},
            }
            for grupo, nome in escolhidos.items()
        },
    }
    treino = Treino(
        doenca=doenca,
        estimativas=estimativas[COLUNAS_ESTIMATIVAS],
        metricas=metricas[COLUNAS_METRICAS],
        modelos=modelos,
        resumo=resumo,
    )
    conferir_qualidade(treino, len(passos))
    return treino


def _cortes_dos_dados() -> dict[str, str]:
    """Data de corte de cada doença no dataset de arboviroses que alimentou o treino."""
    caminho = f"datasets/{get_settings().hf_dataset_repo}/arboviroses/metadados.json"
    doencas = json.loads(HfFileSystem().read_text(caminho, encoding="utf-8"))["doencas"]
    return {doenca: info["corte"] for doenca, info in doencas.items()}


def _validar(estimativas: pd.DataFrame, metricas: pd.DataFrame) -> None:
    # Importados aqui: os contratos usam Polars e Pandera, do grupo `etl`.
    import polars as pl

    from etl.transform.schemas import EstimativaSemanal, MetricaBacktest

    EstimativaSemanal.validate(pl.from_pandas(estimativas))
    MetricaBacktest.validate(pl.from_pandas(metricas))


@dataclass
class Resultado:
    """Tudo o que vai para a pasta `previsao/` do Hub."""

    estimativas: pd.DataFrame
    metricas: pd.DataFrame
    metadados: dict[str, Any]
    modelos: dict[str, dict[str, Any]]  # por doença


def gerar(doencas: list[str]) -> Resultado:
    """Treina as doenças pedidas com as séries do Hub e valida as tabelas."""
    treinos = []
    for doenca in doencas:
        series = dados.ler_series(doenca)
        log.info(
            "%s: %d séries, %d semanas completas.",
            doenca,
            series.casos.shape[1],
            series.semanas_completas,
        )
        treinos.append(treinar(series))

    estimativas = pd.concat([t.estimativas for t in treinos], ignore_index=True)
    metricas = pd.concat([t.metricas for t in treinos], ignore_index=True)
    _validar(estimativas, metricas)
    cortes = _cortes_dos_dados()
    metadados = {
        "tabela": "previsao",
        "linhas": len(estimativas),
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "definicao": "estimativa estatística de tendência dos casos prováveis por semana "
        "epidemiológica de início dos sintomas; não é previsão oficial",
        "semanas_a_frente": SEMANAS_A_FRENTE,
        "criterio_de_municipios": {
            "media_minima_semanal": dados.MEDIA_MINIMA,
            "semanas": dados.SEMANAS_DE_ELEGIBILIDADE,
        },
        "doencas": {t.doenca: {"corte": cortes[t.doenca], **t.resumo} for t in treinos},
        "versoes": {p: version(p) for p in ("scikit-learn", "statsmodels", "pandas", "numpy")},
    }
    return Resultado(
        estimativas=estimativas,
        metricas=metricas,
        metadados=metadados,
        modelos={t.doenca: t.modelos for t in treinos},
    )


def executar(doencas: list[str], publicar: bool = True) -> Resultado:
    """Treina e publica tudo num commit só, a partir de uma pasta temporária do runner."""
    resultado = gerar(doencas)
    log.info("previsao: %s", json.dumps(resultado.metadados, ensure_ascii=False))
    if not publicar:
        log.info("previsao: validado, publicação desativada (--sem-publicar).")
        return resultado

    # Importado aqui para o treino sem publicação não depender do token.
    from etl.publish.hub import publicar_pasta

    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        resultado.estimativas.to_parquet(pasta / "estimativas.parquet", index=False)
        resultado.metricas.to_parquet(pasta / "metricas.parquet", index=False)
        (pasta / "modelos").mkdir()
        for doenca, modelos in resultado.modelos.items():
            joblib.dump(modelos, pasta / "modelos" / f"{doenca}.joblib")
        (pasta / "metadados.json").write_text(
            json.dumps(resultado.metadados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        commit = publicar_pasta(
            pasta,
            "previsao",
            f"previsao: {len(resultado.estimativas)} linhas",
            apagar=["*.parquet", "modelos/*"],
        )
    log.info("previsao: publicado em %s (commit %s).", get_settings().hf_dataset_repo, commit)
    return resultado


def main() -> None:
    """Lê os agregados do Hub, treina, avalia e publica modelo, métricas e estimativas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doencas", nargs="+", choices=dados.DOENCAS, default=dados.DOENCAS)
    parser.add_argument(
        "--sem-publicar", action="store_true", help="Treina e valida sem enviar ao Hub."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # Uma linha por requisição ao Hub polui o log do Actions.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    executar(list(args.doencas), publicar=not args.sem_publicar)


if __name__ == "__main__":
    main()
