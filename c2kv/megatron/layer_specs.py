"""Register the C2KV attention implementation with MCore Bridge."""

from __future__ import annotations

from typing import Optional

from mcore_bridge.model.constant import ModelType
from mcore_bridge.model.register import ModelLoader, ModelMeta, register_model

from .attention import C2KVSelfAttention
from .bridge import C2KVBridge
from .gpt_model import C2KVGPTModel


_GENERIC_MODEL_TYPES = [
    "qwen2",
    "llama",
    "qwen3",
    "qwen2_moe",
    "qwen3_moe",
]


class C2KVModelLoader(ModelLoader):
    model_cls = C2KVGPTModel

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = super().get_transformer_layer_spec(vp_stage)
        for layer_spec in transformer_layer_spec.layer_specs:
            attention_spec = getattr(layer_spec.submodules, "self_attention", None)
            if attention_spec is not None and hasattr(attention_spec, "module"):
                attention_spec.module = C2KVSelfAttention
        return transformer_layer_spec


def register_c2kv_model() -> None:
    register_model(
        ModelMeta(
            ModelType.gpt,
            _GENERIC_MODEL_TYPES,
            bridge_cls=C2KVBridge,
            loader=C2KVModelLoader,
        ),
        exist_ok=True,
    )
