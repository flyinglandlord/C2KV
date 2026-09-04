"""Architecture capabilities used by future lightweight HF adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AttentionCapabilities:
    qk_norm: bool
    qkv_bias: bool
    grouped_query_attention: bool = True


CAPABILITIES: Dict[str, AttentionCapabilities] = {
    "llama": AttentionCapabilities(qk_norm=False, qkv_bias=False),
    "qwen2": AttentionCapabilities(qk_norm=False, qkv_bias=True),
    "qwen3": AttentionCapabilities(qk_norm=True, qkv_bias=False),
}


def capabilities_for(model_type: str) -> AttentionCapabilities:
    try:
        return CAPABILITIES[model_type]
    except KeyError as error:
        raise ValueError(f"unsupported Transformers evaluation model_type: {model_type}") from error
