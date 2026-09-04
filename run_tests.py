#!/usr/bin/env python3
"""Run C2KV tests without installing the repository as a package."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


SUITES = {
    "unit": "tests/unit",
    "parity": "tests/parity",
    "integration": "tests/integration",
    "all": "tests",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="unit")
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / SUITES[args.suite]), pattern=args.pattern, top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
