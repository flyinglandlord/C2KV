"""Dataset adapter hook for the Transformers evaluation pipeline."""

from __future__ import annotations

from typing import Callable, Dict


_LOADERS: Dict[str, Callable] = {}


def register_evaluation_dataset(name: str, loader: Callable) -> None:
    if name in _LOADERS:
        raise ValueError(f"evaluation dataset {name!r} is already registered")
    _LOADERS[name] = loader


def load_evaluation_dataset(name: str, **kwargs):
    if name not in _LOADERS:
        raise ValueError(f"evaluation dataset {name!r} is not registered")
    return _LOADERS[name](**kwargs)
