"""Architecture capability registry for future pretraining model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    hf_model_types: Tuple[str, ...]
    qk_norm: bool
    qkv_bias: bool
    grouped_query_attention: bool = True


_ARCHITECTURES: Dict[str, ArchitectureSpec] = {
    "llama": ArchitectureSpec("llama", ("llama",), qk_norm=False, qkv_bias=False),
    "qwen2": ArchitectureSpec("qwen2", ("qwen2",), qk_norm=False, qkv_bias=True),
    "qwen3": ArchitectureSpec("qwen3", ("qwen3",), qk_norm=True, qkv_bias=False),
}


def register_architecture(spec: ArchitectureSpec) -> None:
    if spec.name in _ARCHITECTURES:
        raise ValueError(f"architecture {spec.name!r} is already registered")
    _ARCHITECTURES[spec.name] = spec


def get_architecture(name: str) -> ArchitectureSpec:
    try:
        return _ARCHITECTURES[name]
    except KeyError as error:
        raise ValueError(f"unknown architecture {name!r}; available: {sorted(_ARCHITECTURES)}") from error
