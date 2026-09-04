"""Stable metric import point while legacy LongBench parity is retained."""

from __future__ import annotations

import sys
from pathlib import Path


def get_legacy_metrics():
    legacy_root = str(Path(__file__).resolve().parents[2] / "python" / "inference")
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    import longbench_metrics

    return longbench_metrics
