"""Pipeline-safe C2KV attention for Megatron Core."""

from __future__ import annotations

import torch
from megatron.core import mpu
from megatron.core.transformer.attention import SelfAttention

from .runtime import get_runtime_settings


class C2KVSelfAttention(SelfAttention):
    """Use a trainable QKV projection only at C2KV memory positions.

    The token selector travels inside the attention-mask dictionary, so activation
    recomputation receives the correct selector for its own micro-batch.
    """

    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        self.c2kv_linear_qkv = submodules.linear_qkv(
            self.config.hidden_size,
            self.linear_qkv_out_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="c2kv_qkv",
            tp_group=self.pg_collection.tp,
        )
        self.c2kv_residual_type = get_runtime_settings().residual_type
        self._c2kv_memory_mask = None
        self._c2kv_compression_source_indices = None

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        memory_mask = None
        if isinstance(attention_mask, dict):
            memory_mask = attention_mask.get("c2kv_memory")
            compression_source_indices = attention_mask.get("c2kv_compression_source_indices")
            attention_mask = attention_mask["full_attention"]
        else:
            compression_source_indices = None
        self._c2kv_memory_mask = memory_mask
        self._c2kv_compression_source_indices = compression_source_indices
        try:
            return super().forward(hidden_states, attention_mask, *args, **kwargs)
        finally:
            self._c2kv_memory_mask = None
            self._c2kv_compression_source_indices = None

    def _local_memory_mask(self, sequence_length: int) -> torch.Tensor:
        memory_mask = self._c2kv_memory_mask
        if memory_mask is None:
            raise RuntimeError("C2KV memory mask is unavailable during QKV projection")
        if memory_mask.shape[1] == sequence_length:
            return memory_mask
        if memory_mask.shape[1] % sequence_length:
            raise ValueError(
                f"cannot align memory mask length {memory_mask.shape[1]} with local sequence {sequence_length}"
            )
        shard_count = memory_mask.shape[1] // sequence_length
        if shard_count != mpu.get_tensor_model_parallel_world_size():
            raise ValueError("unexpected sequence-parallel C2KV mask partition")
        rank = mpu.get_tensor_model_parallel_rank()
        start = rank * sequence_length
        return memory_mask[:, start : start + sequence_length]

    def _memory_projection_input(self, hidden_states):
        if self.c2kv_residual_type == "none":
            return hidden_states
        if self.c2kv_residual_type == "embedding_mean" and self.layer_number != 1:
            return hidden_states
        source_indices = self._c2kv_compression_source_indices
        if source_indices is None or source_indices.shape[1] != hidden_states.shape[0]:
            raise ValueError("C2KV residual sources require an unsharded sequence")
        source_indices = source_indices.to(hidden_states.device)
        valid_sources = source_indices >= 0
        hidden_by_batch = hidden_states.transpose(0, 1)
        batch_indices = torch.arange(hidden_by_batch.shape[0], device=hidden_states.device)[:, None, None]
        source_states = hidden_by_batch[batch_indices, source_indices.clamp_min(0)]
        denominator = valid_sources.sum(dim=-1, keepdim=True).clamp_min(1)
        source_mean = (source_states * valid_sources.unsqueeze(-1)).sum(dim=-2) / denominator
        memory_mask = self._local_memory_mask(hidden_states.shape[0]).transpose(0, 1)
        residual = torch.where(memory_mask.unsqueeze(-1), source_mean.transpose(0, 1), 0)
        return hidden_states + residual

    def get_query_key_value_tensors(self, hidden_states, *args, **kwargs):
        base_outputs = super().get_query_key_value_tensors(hidden_states, *args, **kwargs)
        if self._c2kv_memory_mask is None:
            return base_outputs

        base_projection = self.linear_qkv
        self.linear_qkv = self.c2kv_linear_qkv
        try:
            memory_outputs = super().get_query_key_value_tensors(
                self._memory_projection_input(hidden_states), *args, **kwargs
            )
        finally:
            self.linear_qkv = base_projection

        sequence_length = base_outputs[0].shape[0]
        selector = self._local_memory_mask(sequence_length).transpose(0, 1)
        blended = []
        for base_value, memory_value in zip(base_outputs, memory_outputs):
            if not torch.is_tensor(base_value):
                blended.append(base_value)
                continue
            value_selector = selector
            while value_selector.ndim < base_value.ndim:
                value_selector = value_selector.unsqueeze(-1)
            blended.append(torch.where(value_selector, memory_value, base_value))
        return tuple(blended)

    def _backward_qkv_proj(self):
        super()._backward_qkv_proj()
        if hasattr(self.c2kv_linear_qkv, "backward_dw"):
            self.c2kv_linear_qkv.backward_dw()
