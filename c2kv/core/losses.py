"""Objective weighting kept separate from framework-specific trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class LossWeights:
    language_model: float = 1.0
    reconstruction: float = 0.0
    qkv_regularization: float = 0.0
    self_distillation: float = 0.0

    def combine(self, losses: Mapping[str, object]):
        weighted = None
        for name, weight in (
            ("language_model", self.language_model),
            ("reconstruction", self.reconstruction),
            ("qkv_regularization", self.qkv_regularization),
            ("self_distillation", self.self_distillation),
        ):
            if weight == 0 or name not in losses:
                continue
            term = losses[name] * weight
            weighted = term if weighted is None else weighted + term
        if weighted is None:
            raise ValueError("no enabled loss was provided")
        return weighted
