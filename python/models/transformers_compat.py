"""Compatibility shims for optional Transformers kernel decorators."""

from __future__ import annotations

from inspect import signature

import torch
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask as _create_causal_mask,
    create_sliding_window_causal_mask as _create_sliding_window_causal_mask,
)
from transformers.modeling_rope_utils import rope_config_validation


def auto_docstring(documented_object):
    """Skip version-sensitive docstring synthesis for vendored model code."""

    return documented_object


def configure_rope(config, rope_theta, rope_scaling):
    """Populate the RoPE fields used by both Transformers 4.x and 5.x."""

    if hasattr(config, "standardize_rope_params"):
        config.rope_parameters = rope_scaling
        config.standardize_rope_params()
        config.validate_rope()
        return

    config.rope_scaling = rope_scaling
    if config.rope_scaling is not None and "type" in config.rope_scaling:
        config.rope_scaling["rope_type"] = config.rope_scaling["type"]
    rope_config_validation(config)
    config.rope_parameters = dict(config.rope_scaling or {})
    config.rope_parameters.setdefault("rope_type", "default")
    config.rope_parameters.setdefault("rope_theta", rope_theta)


def validate_layer_types(config):
    """Validate the attention pattern without depending on private APIs."""

    if hasattr(config, "validate_layer_type"):
        config.validate_layer_type()
        return
    if len(config.layer_types) != config.num_hidden_layers:
        raise ValueError("layer_types must contain one entry per hidden layer")
    supported = {"full_attention", "sliding_attention"}
    unknown = set(config.layer_types) - supported
    if unknown:
        raise ValueError(f"unsupported layer types: {sorted(unknown)}")


def _cache_position(input_embeds, past_key_values):
    past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    return torch.arange(
        past_length,
        past_length + input_embeds.shape[1],
        device=input_embeds.device,
    )


def create_causal_mask(
    config,
    input_embeds,
    attention_mask,
    past_key_values=None,
    position_ids=None,
    cache_position=None,
    **kwargs,
):
    call_kwargs = dict(
        past_key_values=past_key_values,
        position_ids=position_ids,
        **kwargs,
    )
    if "cache_position" in signature(_create_causal_mask).parameters:
        call_kwargs["cache_position"] = cache_position if cache_position is not None else _cache_position(
            input_embeds, past_key_values
        )
    return _create_causal_mask(config, input_embeds, attention_mask, **call_kwargs)


def create_sliding_window_causal_mask(
    config,
    input_embeds,
    attention_mask,
    past_key_values=None,
    position_ids=None,
    cache_position=None,
    **kwargs,
):
    call_kwargs = dict(
        past_key_values=past_key_values,
        position_ids=position_ids,
        **kwargs,
    )
    if "cache_position" in signature(_create_sliding_window_causal_mask).parameters:
        call_kwargs["cache_position"] = cache_position if cache_position is not None else _cache_position(
            input_embeds, past_key_values
        )
    return _create_sliding_window_causal_mask(config, input_embeds, attention_mask, **call_kwargs)


try:
    from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
except ImportError:
    from torch import autocast as maybe_autocast

    def merge_with_config_defaults(function):
        return function

try:
    from transformers.utils.output_capturing import capture_outputs
except ImportError:
    def capture_outputs(function):
        return function

try:
    from transformers.integrations import use_kernel_func_from_hub, use_kernelized_func
except ImportError:
    # Transformers 4.57 removed these optional decorators. Their fallback
    # behavior is the undecorated PyTorch implementation already defined in
    # the vendored evaluation models.
    def use_kernel_func_from_hub(_kernel_name):
        return lambda function: function

    def use_kernelized_func(_kernel_function):
        return lambda model_class: model_class


__all__ = [
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernelized_func",
    "maybe_autocast",
    "merge_with_config_defaults",
    "capture_outputs",
    "auto_docstring",
    "configure_rope",
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "validate_layer_types",
]
