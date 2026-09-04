#!/usr/bin/env python3
"""Run the preserved Transformers evaluation pipeline from one JSON config."""

from __future__ import annotations

import argparse
import json

from c2kv.eval import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = run_evaluation(args.config, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
