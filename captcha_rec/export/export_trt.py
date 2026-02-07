import logging
from pathlib import Path

import tensorrt as trt


def export_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = False,
    workspace_mb: int = 1024,
) -> None:
    """
    Convert ONNX to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16: Use FP16 precision
        workspace_mb: Workspace size in MB
    """
    logger = logging.getLogger(__name__)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    logger.info(f"Converting {onnx_path} -> {engine_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape

        min_batch = (1,) + tuple(input_shape[1:])
        opt_batch = (4,) + tuple(input_shape[1:])
        max_batch = (16,) + tuple(input_shape[1:])

        profile.set_shape(input_tensor.name, min_batch, opt_batch, max_batch)

    config.add_optimization_profile(profile)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved: {engine_path}")
