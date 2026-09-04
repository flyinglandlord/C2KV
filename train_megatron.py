#!/usr/bin/env python3
"""Direct-file entry point for C2KV training on Megatron-SWIFT."""

from __future__ import annotations

import argparse
import json

from c2kv.config import load_experiment_config
from c2kv.megatron.arguments import build_ms_swift_arguments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print resolved arguments")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    config = load_experiment_config(cli.config)
    arguments = build_ms_swift_arguments(config)
    if cli.dry_run:
        print(json.dumps({"config": config.to_dict(), "ms_swift_arguments": arguments}, indent=2))
        return
    from c2kv.megatron.model import run_c2kv_megatron

    run_c2kv_megatron(arguments, config)


if __name__ == "__main__":
    main()
