"""Hugging Face Hub helpers: get the data, keep checkpoints safe between sessions."""
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from dmgseg import paths


DATA_PATTERNS = ["Dataset_CVAT/*", "Dataset_CVAT/**", "weights/*"]


def download_data(repo=paths.HF_REPO, local_dir=None, allow_patterns=DATA_PATTERNS):
    """Download images, annotations and the paper weights (not the run checkpoints,
    tens of GB); point dmgseg.paths at them."""
    root = snapshot_download(repo, repo_type="dataset", local_dir=local_dir, allow_patterns=allow_patterns)
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


def safe_upload(local_path, path_in_repo, retries=0, wait=600, log=print):
    """upload() that never stops a training run: failures (e.g. HF's 128 commits
    per hour limit) are logged; with retries > 0 it tries again every `wait` s."""
    import time
    for attempt in range(retries + 1):
        try:
            return upload(local_path, path_in_repo)
        except Exception as e:  # noqa: BLE001 - network / rate limit
            log(f"upload of {path_in_repo} failed ({type(e).__name__}); "
                + (f"retry in {wait} s" if attempt < retries else "giving up"))
            if attempt < retries:
                time.sleep(wait)
    return None


def download_if_exists(path_in_repo, local_dir, repo=paths.HF_REPO):
    """Return the local path of a repo file, or None if it isn't on the Hub."""
    try:
        return Path(hf_hub_download(repo, path_in_repo, repo_type="dataset", local_dir=local_dir))
    except EntryNotFoundError:
        return None


def prior_weights(local_dir=None):
    """Local path of the tool's prior weights (runs/<PRIOR_RUN>/final.pt),
    downloading them from the Hub the first time."""
    local_dir = Path(local_dir or paths.ARTIFACTS)
    target = local_dir / "runs" / paths.PRIOR_RUN / "final.pt"
    if target.exists():
        return target
    path = download_if_exists(f"runs/{paths.PRIOR_RUN}/final.pt", local_dir)
    if path is None:
        raise FileNotFoundError(f"runs/{paths.PRIOR_RUN}/final.pt is not on the Hub")
    return path
