"""Keep legacy full-model copies behind one replaceable evaluation facade."""

from __future__ import annotations

import sys
from pathlib import Path


def _legacy_python_root() -> Path:
    return Path(__file__).resolve().parents[2] / "python"


def load_model_and_tokenizer(model_args, **kwargs):
    """Load the existing C2KV Transformers implementation for evaluation only."""

    legacy_root = str(_legacy_python_root())
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    from models.model_utils import get_model_and_tokenizer

    return get_model_and_tokenizer(model_args, evaluation_mode=True, **kwargs)
