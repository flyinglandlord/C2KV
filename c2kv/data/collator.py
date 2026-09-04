"""Convert framework-neutral LayoutPlan objects into a Megatron batch."""

from __future__ import annotations

from typing import List, Optional, Sequence

from c2kv.core.masking import build_blocked_attention
from c2kv.core.types import IGNORE_INDEX, LayoutPlan, ObjectiveKind, TokenRole


_OBJECTIVE_IDS = {
    ObjectiveKind.IGNORE: 0,
    ObjectiveKind.LANGUAGE_MODEL: 1,
    ObjectiveKind.RECONSTRUCTION: 2,
}


class C2KVCollator:
    def __init__(
        self,
        pad_token_id: int,
        sequence_length: int,
        dense_mask_max_tokens: int,
        include_teacher: bool = False,
    ):
        self.pad_token_id = pad_token_id
        self.sequence_length = sequence_length
        self.dense_mask_max_tokens = dense_mask_max_tokens
        self.include_teacher = include_teacher

    def __call__(self, batch: Sequence[LayoutPlan], padding_to: Optional[int] = None):
        import torch

        if not batch:
            raise ValueError("cannot collate an empty batch")
        longest = max(plan.sequence_length for plan in batch)
        target = max(longest, padding_to or 0)
        target = min(target, self.sequence_length)
        if longest > target:
            raise ValueError(f"compiled sequence length {longest} exceeds configured limit {target}")
        if target > self.dense_mask_max_tokens:
            raise ValueError(
                f"dense reference mask is limited to {self.dense_mask_max_tokens} tokens; "
                "select a registered sparse backend for longer sequences"
            )

        batch_size = len(batch)
        input_ids = torch.full((batch_size, target), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, target), IGNORE_INDEX, dtype=torch.long)
        position_ids = torch.zeros((batch_size, target), dtype=torch.long)
        objective_ids = torch.zeros((batch_size, target), dtype=torch.uint8)
        memory_mask = torch.zeros((batch_size, target), dtype=torch.bool)
        special_token_types = torch.full((batch_size, target), -1, dtype=torch.int8)
        max_compression_sources = max(
            (len(slot.compression_source_indices) for plan in batch for slot in plan.memory_slots),
            default=1,
        )
        compression_source_indices = torch.full(
            (batch_size, target, max_compression_sources), -1, dtype=torch.long
        )
        blocked_attention = torch.ones((batch_size, 1, target, target), dtype=torch.bool)

        for batch_index, plan in enumerate(batch):
            size = plan.sequence_length
            input_ids[batch_index, :size] = torch.tensor(plan.input_ids, dtype=torch.long)
            labels[batch_index, :size] = torch.tensor(plan.labels, dtype=torch.long)
            position_ids[batch_index, :size] = torch.tensor(plan.position_ids, dtype=torch.long)
            objective_ids[batch_index, :size] = torch.tensor(
                [_OBJECTIVE_IDS[value] for value in plan.objective_kinds], dtype=torch.uint8
            )
            memory_mask[batch_index, list(plan.memory_indices)] = True
            special_token_types[batch_index, list(plan.memory_indices)] = 0
            for span in plan.reconstruction_spans:
                special_token_types[batch_index, span.start] = 1
            for slot in plan.memory_slots:
                sources = slot.compression_source_indices
                compression_source_indices[
                    batch_index, slot.sequence_index, : len(sources)
                ] = torch.tensor(sources, dtype=torch.long)
            blocked_attention[batch_index, 0, :size, :size] = torch.tensor(
                build_blocked_attention(plan), dtype=torch.bool
            )
            if size < target:
                padding = torch.arange(size, target)
                blocked_attention[batch_index, 0, padding, padding] = False

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": blocked_attention,
            "c2kv_memory_mask": memory_mask,
            "c2kv_special_token_types": special_token_types,
            "c2kv_compression_source_indices": compression_source_indices,
            "c2kv_objective_ids": objective_ids,
        }
        if self.include_teacher:
            result.update(self._build_teacher_batch(batch, target))
        return result

    def _build_teacher_batch(self, batch: Sequence[LayoutPlan], target: int):
        import torch

        batch_size = len(batch)
        teacher_input_ids = torch.full((batch_size, target), self.pad_token_id, dtype=torch.long)
        teacher_position_ids = torch.zeros((batch_size, target), dtype=torch.long)
        teacher_attention = torch.ones((batch_size, 1, target, target), dtype=torch.bool)
        teacher_index = torch.full((batch_size, target), -1, dtype=torch.long)

        for batch_index, plan in enumerate(batch):
            kept = [
                index
                for index, role in enumerate(plan.token_roles)
                if role
                not in {
                    TokenRole.MEMORY,
                    TokenRole.RECONSTRUCTION_MARKER,
                    TokenRole.RECONSTRUCTION,
                }
            ]
            size = len(kept)
            teacher_input_ids[batch_index, :size] = torch.tensor(
                [plan.input_ids[index] for index in kept], dtype=torch.long
            )
            teacher_position_ids[batch_index, :size] = torch.tensor(
                [plan.position_ids[index] for index in kept], dtype=torch.long
            )
            teacher_attention[batch_index, 0, :size, :size] = torch.triu(
                torch.ones((size, size), dtype=torch.bool), diagonal=1
            )
            for compact_index, original_index in enumerate(kept):
                teacher_index[batch_index, original_index] = compact_index
            if size < target:
                padding = torch.arange(size, target)
                teacher_attention[batch_index, 0, padding, padding] = False
        return {
            "c2kv_teacher_input_ids": teacher_input_ids,
            "c2kv_teacher_position_ids": teacher_position_ids,
            "c2kv_teacher_attention_mask": teacher_attention,
            "c2kv_teacher_index": teacher_index,
        }
