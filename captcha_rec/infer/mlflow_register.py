import logging
from pathlib import Path

import mlflow
import mlflow.onnx
import numpy as np
import onnx

logger = logging.getLogger(__name__)


def register_model(onnx_path) -> None:
    mlflow_tracking_uri = "file:./mlruns"

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    logger.info(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))

    example_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

    experiment_name = "captcha_recognition"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="crnn_onnx_model") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        mlflow.log_param("model_type", "CRNN")
        mlflow.log_param("image_size", 256)
        mlflow.log_param("vocab_size", 63)
        mlflow.log_param("max_len", 20)
        mlflow.log_param("format", "ONNX")

        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            name="model",
            registered_model_name="captcha_crnn",
            input_example=example_input,
        )

        mlflow.log_artifact(str(onnx_path), artifact_path="onnx_files")

        logger.info("Model registered in MLflow")
        logger.info(f"Run ID: {run_id}")
        logger.info("Model available as: captcha_crnn")

    return run_id
