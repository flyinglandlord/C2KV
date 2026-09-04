#!/usr/bin/env python3
"""Export a native C2KV MCore checkpoint to Transformers safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c2kv.config import load_experiment_config
from c2kv.megatron.arguments import build_export_arguments
from c2kv.megatron.checkpoint import write_c2kv_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-precision", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    config = load_experiment_config(cli.config)
    arguments = build_export_arguments(config, cli.checkpoint, cli.output, cli.test_precision)
    if cli.dry_run:
        print(json.dumps(arguments, indent=2))
        return
    from swift.megatron import megatron_export_main

    megatron_export_main(arguments)
    write_c2kv_manifest(cli.output, config)


if __name__ == "__main__":
    main()
