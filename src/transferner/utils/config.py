"""Load YAML configs with ${ENV_VAR} expansion."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def load(name: str) -> dict:
    """load("data") -> configs/data.yaml as a dict, env vars expanded."""
    text = (REPO / "configs" / f"{name}.yaml").read_text()
    return yaml.safe_load(os.path.expandvars(text))


def data_root() -> Path:
    root = os.environ.get("TRANSFERNER_DATA")
    if not root:
        raise RuntimeError("Set TRANSFERNER_DATA to the dataset root on the training box.")
    return Path(root)
