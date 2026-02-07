from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T


def build_transforms(image_size: int) -> Tuple[T.Compose, T.Compose]:
    """
    Returns (train_transforms, eval_transforms).

    Keep it close to original code:
      - ToTensor
      - Resize to (image_size, image_size)
      - Normalize to (-1..1) style via mean=0.5 std=0.5
    """
    base = [
        T.ToTensor(),
        T.Resize((image_size, image_size), antialias=True),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    tfms = T.Compose(base)
    return tfms
