"""Deterministic compression-ratio scheduling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CompressionSchedule:
    ratios: Tuple[int, ...]
    mode: str = "deterministic"
    seed: int = 42

    def ratio_for(self, sample_id: str, epoch: int = 0) -> int:
        if not self.ratios:
            raise ValueError("at least one compression ratio is required")
        if self.mode == "fixed" or len(self.ratios) == 1:
            return self.ratios[0]
        if self.mode != "deterministic":
            raise ValueError(f"unsupported compression schedule: {self.mode}")
        key = f"{self.seed}:{epoch}:{sample_id}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        index = int.from_bytes(digest, byteorder="big") % len(self.ratios)
        return self.ratios[index]
