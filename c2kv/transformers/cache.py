"""Stable import point for C2KV cache concatenation during evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


def blend_memory_key_values(*args, **kwargs):
    legacy_root = str(Path(__file__).resolve().parents[2] / "python")
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    from models.gist_utils import blend_gist_key_values

    return blend_gist_key_values(*args, **kwargs)
