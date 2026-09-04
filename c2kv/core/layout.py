"""Compile paper-level C2KV examples into one pipeline-safe forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .types import (
    IGNORE_INDEX,
    NO_DOCUMENT,
    LayoutPlan,
    MemorySlot,
    ObjectiveKind,
    TokenRole,
    TokenSpan,
    TokenizedExample,
)


@dataclass(frozen=True)
class LayoutCompiler:
    memory_token_id: int
    reconstruction_token_id: int
    max_memory_slots: int = 512
    sink_tokens: int = 0
    overlap_tokens: int = 0

    def compile(self, example: TokenizedExample, compression_ratio: int) -> LayoutPlan:
        example.validate()
        if compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")

        input_ids: List[int] = []
        labels: List[int] = []
        position_ids: List[int] = []
        roles: List[TokenRole] = []
        document_ids: List[int] = []
        objectives: List[ObjectiveKind] = []
        memory_slots: List[MemorySlot] = []
        document_sequence_indices: List[List[int]] = []
        document_memory_indices: List[List[int]] = []

        def append_token(
            token_id: int,
            position_id: int,
            role: TokenRole,
            document_id: int = NO_DOCUMENT,
            label: int = IGNORE_INDEX,
            objective: ObjectiveKind = ObjectiveKind.IGNORE,
        ) -> int:
            index = len(input_ids)
            input_ids.append(token_id)
            labels.append(label)
            position_ids.append(position_id)
            roles.append(role)
            document_ids.append(document_id)
            objectives.append(objective)
            return index

        for position, token_id in enumerate(example.system_tokens):
            append_token(token_id, position, TokenRole.SYSTEM)

        logical_document_offset = len(example.system_tokens)
        effective_sink_tokens = self.sink_tokens or compression_ratio
        for document_id, document in enumerate(example.document_tokens):
            raw_indices: List[int] = []
            memory_indices: List[int] = []
            for chunk_id, source_start in enumerate(range(0, len(document), compression_ratio)):
                source_end = min(source_start + compression_ratio, len(document))
                for local_index in range(source_start, source_end):
                    raw_indices.append(
                        append_token(
                            document[local_index],
                            logical_document_offset + local_index,
                            TokenRole.DOCUMENT,
                            document_id,
                        ))

                if len(memory_slots) >= self.max_memory_slots:
                    raise ValueError(
                        f"sample {example.sample_id!r} needs more than {self.max_memory_slots} memory slots"
                    )
                overlap_start = max(0, source_start - self.overlap_tokens)
                local_sources = set(range(0, min(effective_sink_tokens, len(document))))
                local_sources.update(range(overlap_start, source_end))
                visible_document_indices = tuple(
                    raw_indices[i] for i in sorted(local_sources) if i < len(raw_indices)
                )
                compression_source_indices = tuple(raw_indices[source_start:source_end])
                memory_index = append_token(
                    self.memory_token_id,
                    logical_document_offset + source_end - 1,
                    TokenRole.MEMORY,
                    document_id,
                )
                memory_slots.append(
                    MemorySlot(
                        sequence_index=memory_index,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        source_start=source_start,
                        source_end=source_end,
                        compression_source_indices=compression_source_indices,
                        visible_document_indices=visible_document_indices,
                        previous_memory_indices=tuple(memory_indices),
                        logical_position=logical_document_offset + source_end - 1,
                    ))
                memory_indices.append(memory_index)

            document_sequence_indices.append(raw_indices)
            document_memory_indices.append(memory_indices)
            logical_document_offset += len(document)

        response_logical_start = logical_document_offset
        for offset, token_id in enumerate(example.query_tokens):
            append_token(token_id, response_logical_start + offset, TokenRole.QUERY)
        response_start = len(input_ids)
        for offset, token_id in enumerate(example.response_tokens):
            append_token(
                token_id,
                response_logical_start + len(example.query_tokens) + offset,
                TokenRole.RESPONSE,
                label=token_id,
                objective=ObjectiveKind.LANGUAGE_MODEL,
            )
        response_span = None
        if example.response_tokens:
            response_span = TokenSpan(response_start, len(input_ids), TokenRole.RESPONSE)

        reconstruction_spans: List[TokenSpan] = []
        reconstruction_position = response_logical_start + len(example.query_tokens) + len(example.response_tokens)
        for document_id in example.reconstruction_document_ids:
            span_start = len(input_ids)
            append_token(
                self.reconstruction_token_id,
                reconstruction_position,
                TokenRole.RECONSTRUCTION_MARKER,
                document_id,
            )
            reconstruction_position += 1
            for token_id in example.document_tokens[document_id]:
                append_token(
                    token_id,
                    reconstruction_position,
                    TokenRole.RECONSTRUCTION,
                    document_id,
                    label=token_id,
                    objective=ObjectiveKind.RECONSTRUCTION,
                )
                reconstruction_position += 1
            reconstruction_spans.append(
                TokenSpan(span_start, len(input_ids), TokenRole.RECONSTRUCTION, document_id)
            )

        plan = LayoutPlan(
            sample_id=example.sample_id,
            input_ids=tuple(input_ids),
            labels=tuple(labels),
            position_ids=tuple(position_ids),
            token_roles=tuple(roles),
            document_ids=tuple(document_ids),
            objective_kinds=tuple(objectives),
            memory_slots=tuple(memory_slots),
            response_span=response_span,
            reconstruction_spans=tuple(reconstruction_spans),
        )
        plan.validate()
        return plan
