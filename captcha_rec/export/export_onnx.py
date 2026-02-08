from __future__ import annotations

import logging
from pathlib import Path

import torch

from captcha_rec.models.lightning_module import LACCModule

logger = logging.getLogger(__name__)


def export_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    image_size: int,
    vocab_size: int,
    max_len: int,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    lit: LACCModule = LACCModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        vocab_size=vocab_size,
        max_len=max_len,
        lr=1e-4,
        weight_decay=1e-2,
        pad_id=0,
        optimizer_name="lion",
    )
    lit.eval()
    model = lit.model.eval().cpu()

    dummy = torch.randn(
        1,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
        device="cpu",
    )

    logger.info("Exporting ONNX to: %s", onnx_path)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    logger.info("ONNX exported successfully.")
