"""Process-local C2KV settings consumed while MCore constructs the model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class C2KVRuntimeSettings:
    residual_type: str = "none"


_SETTINGS = C2KVRuntimeSettings()


def configure_runtime(*, residual_type: str) -> None:
    global _SETTINGS
    _SETTINGS = C2KVRuntimeSettings(residual_type=residual_type)


def get_runtime_settings() -> C2KVRuntimeSettings:
    return _SETTINGS
