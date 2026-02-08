from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

from captcha_rec.data.datamodule import build_default_vocab
from captcha_rec.infer.schema import Prediction

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _preprocess_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0

    arr = (arr - 0.5) / 0.5

    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def decode_tokens(token_ids: Sequence[int], pad_token: str = "<pad>") -> str:
    vocab = build_default_vocab()
    chars = []
    for tid in token_ids:
        ch = vocab.id_to_token.get(int(tid), "?")
        if ch == pad_token:
            continue
        chars.append(ch)
    return "".join(chars)


def iter_image_paths(inputs: Sequence[str]) -> Iterable[Path]:
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                yield from p.glob(ext)
        else:
            yield p


def run_onnx_infer(
    onnx_path: Path,
    inputs: Sequence[str],
    output_jsonl: Path,
    image_size: int,
) -> None:
    onnx_path = Path(onnx_path)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    logger.info("Loading ONNX: %s", onnx_path)
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    all_preds: list[Prediction] = []
    for img_path in iter_image_paths(inputs):
        if not img_path.exists():
            logger.warning("Skip missing: %s", img_path)
            continue

        x = _preprocess_image(img_path, image_size=image_size)
        logits = sess.run(["logits"], {"input": x})[0]
        probs = _softmax(logits, axis=1)
        token_ids = np.argmax(probs, axis=1)[0].tolist()
        pred_text = decode_tokens(token_ids)

        all_preds.append(Prediction(path=str(img_path), pred_text=pred_text))

    logger.info("Writing predictions to: %s", output_jsonl)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for pr in all_preds:
            f.write(
                json.dumps(
                    {"path": pr.path, "pred_text": pr.pred_text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info("Done. Total predictions: %d", len(all_preds))
