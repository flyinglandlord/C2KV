"""Reference mask semantics and a sparse range interface for future kernels."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .types import LayoutPlan, TokenRole


IndexRange = Tuple[int, int]


def _reconstruction_span_start(plan: LayoutPlan, query_index: int) -> int:
    for span in plan.reconstruction_spans:
        if span.contains(query_index):
            return span.start
    raise ValueError(f"reconstruction token {query_index} is not covered by a span")


def allowed_key_indices(plan: LayoutPlan, query_index: int) -> Tuple[int, ...]:
    """Return keys visible to one query token under C2KV semantics."""

    role = plan.token_roles[query_index]
    document_id = plan.document_ids[query_index]
    system = [i for i, r in enumerate(plan.token_roles) if r == TokenRole.SYSTEM]

    if role == TokenRole.PADDING:
        return (query_index,)
    if role == TokenRole.SYSTEM:
        return tuple(i for i in system if i <= query_index)
    if role == TokenRole.DOCUMENT:
        document_prefix = [
            i
            for i, r in enumerate(plan.token_roles)
            if r == TokenRole.DOCUMENT and plan.document_ids[i] == document_id and i <= query_index
        ]
        return tuple(system + document_prefix)
    if role == TokenRole.MEMORY:
        slot = next(slot for slot in plan.memory_slots if slot.sequence_index == query_index)
        return tuple(
            sorted(
                set(
                    system
                    + list(slot.visible_document_indices)
                    + list(slot.previous_memory_indices)
                    + [query_index]
                )
            )
        )
    if role in {TokenRole.QUERY, TokenRole.RESPONSE}:
        memory = list(plan.memory_indices)
        causal_tail = [
            i
            for i, r in enumerate(plan.token_roles)
            if r in {TokenRole.QUERY, TokenRole.RESPONSE} and i <= query_index
        ]
        return tuple(system + memory + causal_tail)
    if role in {TokenRole.RECONSTRUCTION_MARKER, TokenRole.RECONSTRUCTION}:
        span_start = _reconstruction_span_start(plan, query_index)
        memory = [
            i
            for i in plan.memory_indices
            if plan.document_ids[i] == document_id
        ]
        causal_reconstruction = list(range(span_start, query_index + 1))
        return tuple(system + memory + causal_reconstruction)
    raise ValueError(f"unsupported token role: {role}")


def iter_allowed_key_ranges(plan: LayoutPlan, query_index: int) -> Iterable[IndexRange]:
    """Yield half-open ranges consumed by a future block-sparse backend."""

    indices = allowed_key_indices(plan, query_index)
    if not indices:
        return
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        yield start, previous + 1
        start = previous = index
    yield start, previous + 1


def build_allowed_attention(plan: LayoutPlan) -> Tuple[Tuple[bool, ...], ...]:
    """Materialize the dense reference mask; True means attention is allowed."""

    size = plan.sequence_length
    rows: List[Tuple[bool, ...]] = []
    for query_index in range(size):
        allowed = set(allowed_key_indices(plan, query_index))
        rows.append(tuple(key_index in allowed for key_index in range(size)))
    return tuple(rows)


def build_blocked_attention(plan: LayoutPlan) -> Tuple[Tuple[bool, ...], ...]:
    """Materialize the Megatron convention; True means attention is blocked."""

    return tuple(tuple(not value for value in row) for row in build_allowed_attention(plan))
