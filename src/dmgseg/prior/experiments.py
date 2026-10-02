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
from dmgseg.prior.compare import checkpoint_spec, compare_checkpoints, results_table
from dmgseg.prior.train import train

_threads = globals().get("_threads") or {}  # survives importlib.reload


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


def run_queue(queue, config_paths, workdir, embed_model, embed_fn, device, baselines, wait_for=None):
    """Train each config (skipping runs whose final.pt is already on the Hub), then
    compare baselines + new final.pt files on full validation images, each at its
    own training scale. wait_for: name of another background job to finish first."""
    workdir = Path(workdir)
    finals = {}
    try:
        if wait_for and is_running(wait_for):
            _status(workdir, queue, state="waiting", stage=f"for {wait_for} to finish")
            _find(wait_for).join()
        for cfg_path in config_paths:
            cfg = load_config(cfg_path)
            run_dir = workdir / "runs" / cfg.run_name
            if _final_on_hub(cfg.run_name):
                hub.download_if_exists(f"runs/{cfg.run_name}/final.pt", workdir)
                print(f"{cfg.run_name}: already trained")
            else:
                _status(workdir, queue, state="training", run=cfg.run_name, stage=cfg.run_name)
                train(cfg, workdir, embed_fn, device=device)
            finals[cfg.run_name] = checkpoint_spec(cfg, run_dir / "final.pt")

        _status(workdir, queue, state="comparing")
        out = workdir / "runs" / queue / "comparison.json"
        results = compare_checkpoints({**baselines, **finals}, embed_model, device, out)
        hub.upload(out, f"runs/{queue}/comparison.json")
        _status(workdir, queue, state="done", table=results_table(results, "fixed_labels"))
    except Exception:
        _status(workdir, queue, state="error", error=traceback.format_exc()[-3000:])
        raise


def start_in_background(queue, *args, target=None, wait_for=None, **kwargs):
    """Start target (default run_queue(queue, ...)) in a daemon thread unless a job
    with this name is already running. wait_for: name of a job to finish first
    (works for any target; a `status` keyword argument, if given, is told)."""
    if is_running(queue):
        return f"{queue}: already running"
    if target is None:
        target, args = run_queue, (queue, *args)
    job = target

    def runner(*a, **kw):
        _quiet_progress_bars()
        if wait_for and is_running(wait_for):
            if callable(kw.get("status")):
                kw["status"](state="waiting", stage=f"for {wait_for} to finish")
            _find(wait_for).join()
        return job(*a, **kw)

    t = threading.Thread(target=runner, args=args, kwargs=kwargs, name=queue, daemon=True)
    t.start()
    _threads[queue] = t
    return f"{queue}: started"


def _find(queue):
    t = _threads.get(queue)
    if t is None:  # e.g. started before a reload: look it up by thread name
        t = next((x for x in threading.enumerate() if x.name == queue), None)
    return t


def _quiet_progress_bars():
    """Background threads must not write to stdout/stderr: in a marimo kernel a thread
    that is not attached to a running cell has no output stream (AssertionError in
    marimo's stream). Progress goes to the job's status file instead."""
    import tqdm as _tqdm_mod
    cls = _tqdm_mod.tqdm
    # A fresh write lock per job: a thread that crashed while holding tqdm's shared
    # lock (it happened: a marimo stream error inside tqdm.write) would otherwise
    # block every later progress bar in this process forever.
    from tqdm.std import TRLock, TqdmDefaultWriteLock
    TqdmDefaultWriteLock.th_lock = TRLock()           # the lock instances actually share
    if hasattr(TqdmDefaultWriteLock, "mp_lock"):
        del TqdmDefaultWriteLock.mp_lock              # recreated by the next instance
    cls.set_lock(TqdmDefaultWriteLock())
    if not getattr(cls, "_dmgseg_quiet", False):
        original = cls.__init__

        def init(self, *args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                kwargs["disable"] = True
            original(self, *args, **kwargs)
        cls.__init__ = init
        cls._dmgseg_quiet = True
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass


def is_running(queue):
    t = _find(queue)
    return t is not None and t.is_alive()
