from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    path: str
    pred_text: str
