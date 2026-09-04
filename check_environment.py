#!/usr/bin/env python3
"""Check the external runtime used by the direct-file C2KV workflow."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from c2kv.config import load_experiment_config


PACKAGES = {
    "torch": ("torch", True),
    "transformers": ("transformers", True),
    "ms-swift": ("swift", True),
    "megatron-core": ("megatron.core", True),
    "mcore-bridge": ("mcore_bridge", True),
    "safetensors": ("safetensors", True),
    "wandb": ("wandb", False),
}


def package_status(distribution: str, module: str, required: bool):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return {"required": required, "available": False, "version": None, "module": module}
    try:
        importlib.import_module(module)
    except Exception as error:
        return {
            "required": required,
            "available": False,
            "version": version,
            "module": module,
            "import_error": f"{type(error).__name__}: {error}",
        }
    return {"required": required, "available": True, "version": version, "module": module}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-wandb", action="store_true")
    args = parser.parse_args()
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: package_status(distribution, module, required)
            for distribution, (module, required) in PACKAGES.items()
            for name in [distribution]
        },
    }
    if args.config:
        config = load_experiment_config(args.config)
        result["config"] = {"valid": True, "model": config.model.name_or_path}
    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except ImportError:
        result["cuda"] = {"available": False, "device_count": 0, "devices": []}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    missing = [name for name, status in result["packages"].items() if status["required"] and not status["available"]]
    if args.require_wandb and not result["packages"]["wandb"]["available"]:
        missing.append("wandb")
    if args.strict and missing:
        raise SystemExit(f"missing required runtime packages: {', '.join(missing)}")


if __name__ == "__main__":
    main()
