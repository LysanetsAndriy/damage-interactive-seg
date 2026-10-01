"""Hugging Face Hub helpers: get the data, keep checkpoints safe between sessions."""
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from dmgseg import paths


def download_data(repo=paths.HF_REPO, local_dir=None):
    """Download images, annotations and weights; point dmgseg.paths at them.

    Call this before importing anything that reads dmgseg.paths at import time.
    """
    root = snapshot_download(repo, repo_type="dataset", local_dir=local_dir)
    os.environ["DMGSEG_DATA"] = root
    _refresh_paths()
    return Path(root)


def _refresh_paths():
    import importlib
    importlib.reload(paths)


def upload(local_path, path_in_repo, repo=paths.HF_REPO, message=None):
    HfApi().upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                        repo_id=repo, repo_type="dataset",
                        commit_message=message or f"Upload {path_in_repo}")


def download_if_exists(path_in_repo, local_dir, repo=paths.HF_REPO):
    """Return the local path of a repo file, or None if it isn't on the Hub."""
    try:
        return Path(hf_hub_download(repo, path_in_repo, repo_type="dataset", local_dir=local_dir))
    except EntryNotFoundError:
        return None
