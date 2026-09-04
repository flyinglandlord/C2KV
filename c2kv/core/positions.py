"""RoPE position utilities shared by training and evaluation."""

from __future__ import annotations

from typing import Iterable, Tuple


def concatenate_memory_positions(
    document_memory_positions: Iterable[Iterable[int]],
    document_lengths: Iterable[int],
    prefix_length: int = 0,
) -> Tuple[int, ...]:
    """Shift independently-prefilled memory positions into concatenated documents."""

    positions = []
    offset = prefix_length
    for local_positions, document_length in zip(document_memory_positions, document_lengths):
        positions.extend(offset + position for position in local_positions)
        offset += document_length
    return tuple(positions)
