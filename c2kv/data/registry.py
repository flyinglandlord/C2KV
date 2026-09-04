"""Small adapter registry for adding paper datasets without trainer changes."""

from __future__ import annotations

from typing import Callable, Dict, Mapping

from .schema import C2KVExample


Adapter = Callable[[Mapping[str, object], str], C2KVExample]
_ADAPTERS: Dict[str, Adapter] = {"default": C2KVExample.from_mapping}


def register_adapter(name: str, adapter: Adapter) -> None:
    if not name or name in _ADAPTERS:
        raise ValueError(f"adapter {name!r} is empty or already registered")
    _ADAPTERS[name] = adapter


def get_adapter(name: str) -> Adapter:
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"unknown dataset adapter {name!r}; available: {sorted(_ADAPTERS)}") from error
