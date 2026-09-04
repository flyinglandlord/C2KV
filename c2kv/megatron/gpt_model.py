"""MCore GPT model extension for paper-level C2KV special embeddings."""

from __future__ import annotations

import torch
from megatron.core import mpu
from mcore_bridge.model import GPTModel


class C2KVGPTModel(GPTModel):
    """Replace memory and reconstruction markers with two trainable embeddings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pre_process:
            self.c2kv_special_embeddings = torch.nn.Parameter(
                self.embedding.word_embeddings.weight.new_zeros(2, self.config.hidden_size)
            )
            setattr(
                self.c2kv_special_embeddings,
                "sequence_parallel",
                self.config.sequence_parallel,
            )
            setattr(self.c2kv_special_embeddings, "tensor_model_parallel", False)
        self._c2kv_special_token_types = None

    def forward(self, input_ids, position_ids, attention_mask=None, *args, **kwargs):
        special_token_types = None
        if isinstance(attention_mask, dict):
            special_token_types = attention_mask.get("c2kv_special_token_types")
        self._c2kv_special_token_types = special_token_types
        try:
            return super().forward(input_ids, position_ids, attention_mask, *args, **kwargs)
        finally:
            self._c2kv_special_token_types = None

    def _local_special_token_types(self, sequence_length: int):
        special_token_types = self._c2kv_special_token_types
        if special_token_types is None or special_token_types.shape[1] == sequence_length:
            return special_token_types
        if special_token_types.shape[1] % sequence_length:
            raise ValueError("cannot align C2KV special-token types with the local sequence")
        shard_count = special_token_types.shape[1] // sequence_length
        if shard_count != mpu.get_tensor_model_parallel_world_size():
            raise ValueError("unexpected sequence-parallel C2KV embedding partition")
        rank = mpu.get_tensor_model_parallel_rank()
        start = rank * sequence_length
        return special_token_types[:, start : start + sequence_length]

    def _replace_special_embeddings(self, hidden_states):
        if hidden_states is None or self._c2kv_special_token_types is None:
            return hidden_states
        special_token_types = self._local_special_token_types(hidden_states.shape[0])
        special_token_types = special_token_types.transpose(0, 1)
        valid = special_token_types >= 0
        replacement = self.c2kv_special_embeddings[special_token_types.clamp_min(0)]
        return torch.where(valid.unsqueeze(-1), replacement, hidden_states)

    def _preprocess(self, *args, **kwargs):
        outputs = list(super()._preprocess(*args, **kwargs))
        original_decoder_input = outputs[0]
        decoder_input = self._replace_special_embeddings(original_decoder_input)
        outputs[0] = decoder_input
        if outputs[1] is original_decoder_input:
            outputs[1] = decoder_input
        return tuple(outputs)
