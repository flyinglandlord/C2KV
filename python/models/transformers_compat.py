"""Compatibility shims for optional Transformers kernel decorators."""

from __future__ import annotations

from transformers.integrations import use_kernel_forward_from_hub

try:
    from transformers.integrations import use_kernel_func_from_hub, use_kernelized_func
except ImportError:
    # Transformers 4.57 removed these optional decorators. Their fallback
    # behavior is the undecorated PyTorch implementation already defined in
    # the vendored evaluation models.
    def use_kernel_func_from_hub(_kernel_name):
        return lambda function: function

    def use_kernelized_func(function):
        return function


__all__ = [
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernelized_func",
]
