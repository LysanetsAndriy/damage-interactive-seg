"""YAML configs loaded as nested attribute dicts (cfg.train.lr)."""
from pathlib import Path

import yaml


class Cfg(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(value) if isinstance(value, dict) else value


def load_config(path, **overrides):
    """Load a YAML config. Overrides use dotted keys: load_config(p, **{"train.epochs": 1})."""
    cfg = yaml.safe_load(Path(path).read_text())
    for dotted, value in overrides.items():
        node = cfg
        *parents, last = dotted.split(".")
        for p in parents:
            node = node[p]
        node[last] = value
    return Cfg(cfg)
