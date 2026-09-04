"""Attention-mask backend registry.

The dense backend is the correctness oracle. Production block-sparse kernels can
register here and consume `iter_allowed_key_ranges` without changing data semantics.
"""

from __future__ import annotations

from typing import Callable, Dict


_BACKENDS: Dict[str, Callable] = {}


def register_attention_backend(name: str, builder: Callable) -> None:
    if not name or name in _BACKENDS:
        raise ValueError(f"attention backend {name!r} is empty or already registered")
    _BACKENDS[name] = builder


def get_attention_backend(name: str) -> Callable:
    try:
        return _BACKENDS[name]
    except KeyError as error:
        raise ValueError(f"attention backend {name!r} is not registered") from error
