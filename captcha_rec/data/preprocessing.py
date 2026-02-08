from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T


def build_transforms(image_size: int) -> Tuple[T.Compose, T.Compose]:
    base = [
        T.ToTensor(),
        T.Resize((image_size, image_size), antialias=True),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    tfms = T.Compose(base)
    return tfms
