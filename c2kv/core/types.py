"""Names used by both Megatron training and Transformers evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Tuple


IGNORE_INDEX = -100
NO_DOCUMENT = -1


class TokenRole(IntEnum):
    """Semantic role of each token in the paper's structured forward pass."""

    PADDING = 0
    SYSTEM = 1
    DOCUMENT = 2
    MEMORY = 3
    QUERY = 4
    RESPONSE = 5
    RECONSTRUCTION_MARKER = 6
    RECONSTRUCTION = 7


class ObjectiveKind(str, Enum):
    IGNORE = "ignore"
    LANGUAGE_MODEL = "language_model"
    RECONSTRUCTION = "reconstruction"


@dataclass(frozen=True)
class TokenizedExample:
    sample_id: str
    system_tokens: Tuple[int, ...]
    document_tokens: Tuple[Tuple[int, ...], ...]
    query_tokens: Tuple[int, ...]
    response_tokens: Tuple[int, ...]
    reconstruction_document_ids: Tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.document_tokens:
            raise ValueError("at least one document is required")
        if any(not document for document in self.document_tokens):
            raise ValueError("documents must not be empty")
        if any(i < 0 or i >= len(self.document_tokens) for i in self.reconstruction_document_ids):
            raise ValueError("reconstruction_document_ids contains an invalid document index")


@dataclass(frozen=True)
class MemorySlot:
    """One C2KV slot and the source tokens it compresses."""

    sequence_index: int
    document_id: int
    chunk_id: int
    source_start: int
    source_end: int
    compression_source_indices: Tuple[int, ...]
    visible_document_indices: Tuple[int, ...]
    previous_memory_indices: Tuple[int, ...]
    logical_position: int


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    role: TokenRole
    document_id: int = NO_DOCUMENT

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end


@dataclass(frozen=True)
class LayoutPlan:
    """Complete, framework-neutral description of one C2KV forward pass."""

    sample_id: str
    input_ids: Tuple[int, ...]
    labels: Tuple[int, ...]
    position_ids: Tuple[int, ...]
    token_roles: Tuple[TokenRole, ...]
    document_ids: Tuple[int, ...]
    objective_kinds: Tuple[ObjectiveKind, ...]
    memory_slots: Tuple[MemorySlot, ...]
    response_span: Optional[TokenSpan]
    reconstruction_spans: Tuple[TokenSpan, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    @property
    def memory_indices(self) -> Tuple[int, ...]:
        return tuple(slot.sequence_index for slot in self.memory_slots)

    def validate(self) -> None:
        size = self.sequence_length
        aligned = (
            self.labels,
            self.position_ids,
            self.token_roles,
            self.document_ids,
            self.objective_kinds,
        )
        if any(len(values) != size for values in aligned):
            raise ValueError("all token-aligned LayoutPlan fields must have equal length")
        if any(slot.sequence_index >= size for slot in self.memory_slots):
            raise ValueError("memory slot points outside the sequence")
