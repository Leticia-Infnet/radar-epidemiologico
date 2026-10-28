"""Publicação de artefatos (Parquet, modelo, índice) no Hugging Face Hub."""

from pathlib import Path

from huggingface_hub import HfApi

from common.config import Settings, get_settings


def publicar_pasta(
    pasta_local: Path,
    caminho_no_repo: str,
    mensagem: str,
    settings: Settings | None = None,
) -> str:
    """Envia uma pasta gerada no runner do CI para o repositório *dataset* do Hub.

    A pasta local é temporária: existe só durante o job do GitHub Actions.
    """
    settings = settings or get_settings()
    if settings.hf_token is None:
        raise RuntimeError("Defina RADAR_HF_TOKEN para publicar no Hub.")
    api = HfApi(token=settings.hf_token.get_secret_value())
    api.create_repo(settings.hf_dataset_repo, repo_type="dataset", exist_ok=True)
    commit = api.upload_folder(
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        folder_path=pasta_local,
        path_in_repo=caminho_no_repo,
        commit_message=mensagem,
    )
    return commit.oid
