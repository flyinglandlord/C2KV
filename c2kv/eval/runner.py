"""JSON-configured evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .backends import run_backend


def load_evaluation_config(path: str) -> Dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a JSON object")
    for key in ("model", "dataset", "output_file"):
        if not config.get(key):
            raise ValueError(f"evaluation config requires {key!r}")
    return config


def run_evaluation(config_path: str, dry_run: bool = False) -> List[str]:
    return run_backend(load_evaluation_config(config_path), dry_run=dry_run)
