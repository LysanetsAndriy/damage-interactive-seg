"""Run a queue of prior-training configs in a background thread, then compare them.

A background thread keeps the training alive even if the notebook cell that
started it is interrupted (e.g. when an agent's connection drops). Progress is
visible on the Hub: runs/<run>/history.json per epoch, and runs/<queue>/status.json.
"""
import json
import threading
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi

from dmgseg import hub, paths
from dmgseg.config import load_config
from dmgseg.prior.compare import compare_checkpoints, results_table
from dmgseg.prior.train import train

_threads = {}


def status_writer(workdir, queue):
    """-> status(**fields) that updates runs/<queue>/status.json locally and on the Hub."""
    return lambda **fields: _status(workdir, queue, **fields)


def _status(workdir, queue, **fields):
    path = Path(workdir) / "runs" / queue / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(fields, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    path.write_text(json.dumps(state, indent=1))
    try:
        hub.upload(path, f"runs/{queue}/status.json")
    except Exception as e:  # status is best-effort; never kill training over it
        print("status upload failed:", e)


def _final_on_hub(run_name):
    return HfApi().file_exists(paths.HF_REPO, f"runs/{run_name}/final.pt", repo_type="dataset")


def run_queue(queue, config_paths, workdir, embed_model, embed_fn, device, baselines):
    """Train each config (skipping runs whose final.pt is already on the Hub), then
    compare baselines + new final.pt files on full validation images."""
    workdir = Path(workdir)
    finals = {}
    try:
        for cfg_path in config_paths:
            cfg = load_config(cfg_path)
            run_dir = workdir / "runs" / cfg.run_name
            if _final_on_hub(cfg.run_name):
                hub.download_if_exists(f"runs/{cfg.run_name}/final.pt", workdir)
                print(f"{cfg.run_name}: already trained")
            else:
                _status(workdir, queue, state="training", run=cfg.run_name)
                train(cfg, workdir, embed_fn, device=device)
            finals[cfg.run_name] = run_dir / "final.pt"

        _status(workdir, queue, state="comparing")
        out = workdir / "runs" / queue / "comparison.json"
        results = compare_checkpoints({**baselines, **finals}, embed_model, device, out)
        hub.upload(out, f"runs/{queue}/comparison.json")
        _status(workdir, queue, state="done", table=results_table(results, "fixed_labels"))
    except Exception:
        _status(workdir, queue, state="error", error=traceback.format_exc()[-3000:])
        raise


def start_in_background(queue, *args, target=None, **kwargs):
    """Start target (default run_queue(queue, ...)) in a daemon thread unless a job
    with this name is already running."""
    t = _threads.get(queue)
    if t is not None and t.is_alive():
        return f"{queue}: already running"
    if target is None:
        target, args = run_queue, (queue, *args)
    t = threading.Thread(target=target, args=args, kwargs=kwargs, name=queue, daemon=True)
    t.start()
    _threads[queue] = t
    return f"{queue}: started"


def is_running(queue):
    t = _threads.get(queue)
    return t is not None and t.is_alive()
