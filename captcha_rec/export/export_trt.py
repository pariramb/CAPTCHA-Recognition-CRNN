import logging
import subprocess
from pathlib import Path
from typing import Optional


def export_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = False,
    workspace_mb: int = 1024,
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 16,
    image_size: Optional[int] = None,
) -> None:
    logger = logging.getLogger(__name__)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    logger.info(f"Converting {onnx_path} -> {engine_path}")

    image_size = image_size or 256

    cmd = [
        "polygraphy",
        "convert",
        str(onnx_path),
        "--output",
        str(engine_path),
    ]

    if fp16:
        cmd.append("--fp16")

    cmd.extend(
        [
            "--trt-min-shapes",
            f"input:[{min_batch},3,{image_size},{image_size}]",
            "--trt-opt-shapes",
            f"input:[{opt_batch},3,{image_size},{image_size}]",
            "--trt-max-shapes",
            f"input:[{max_batch},3,{image_size},{image_size}]",
        ]
    )

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = f"Polygraphy failed: {result.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not engine_path.exists():
        raise RuntimeError("TensorRT engine was not created")

    logger.info(f"TensorRT engine created: {engine_path}")
