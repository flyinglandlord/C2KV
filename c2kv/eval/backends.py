"""Evaluation backend registry; training code never imports this module."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List


_SCRIPTS = {
    "c2kv": "expr_c2kv.py",
    "full_compute": "expr_fullcompute.py",
    "reuse": "expr_reuse.py",
    "block_attention": "expr_blockattention.py",
    "cacheblend": "expr_cacheblend.py",
}


def build_legacy_command(config: Dict[str, object]) -> List[str]:
    backend = str(config.get("backend", "c2kv"))
    try:
        script_name = _SCRIPTS[backend]
    except KeyError as error:
        raise ValueError(f"unknown evaluation backend {backend!r}; available: {sorted(_SCRIPTS)}") from error
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "python" / "inference" / script_name)]
    mapping = {
        "model": "--model",
        "dataset": "--dataset",
        "dataset_path": "--dataset_path",
        "output_file": "--output_file",
        "max_examples": "--max_examples",
        "cut_length": "--cut_length",
    }
    for key, flag in mapping.items():
        value = config.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    for flag in ("profile", "only_supporting", "cot"):
        if config.get(flag):
            command.append(f"--{flag}")
    for value in config.get("extra_args", []):
        command.append(str(value))
    return command


def run_backend(config: Dict[str, object], dry_run: bool = False) -> List[str]:
    command = build_legacy_command(config)
    if not dry_run:
        subprocess.run(command, check=True)
    return command
