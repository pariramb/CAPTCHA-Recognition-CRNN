#!/usr/bin/env python3
"""Script to export trained model to different formats."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).parent.parent))
from src.models.lacc import LACC


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def export_model(config: DictConfig):
    """Export model to different formats."""
    
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript", "all"], 
                       default="all", help="Export format")
    parser.add_argument("--model-path", type=str, 
                       default="models/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, 
                       default="exported_models", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = OmegaConf.to_container(config, resolve=True)
    parser_args = parser.parse_args()
    
    model_path = Path(parser_args.model_path)
    output_dir = Path(parser_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=parser_args.device)
    
    model = LACC(config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(parser_args.device)
    
    sample_input = torch.randn(
        1, 3, config.data.image_size, config.data.image_size
    ).to(parser_args.device)
    
    if parser_args.format in ["onnx", "all"]:
        export_to_onnx(
            model=model,
            sample_input=sample_input,
            output_path=output_dir / "model.onnx",
            config=config,
            token_dict=checkpoint.get("token_dict"),
        )
    
    if parser_args.format in ["torchscript", "all"]:
        export_to_torchscript(
            model=model,
            sample_input=sample_input,
            output_path=output_dir / "model.pt",
        )
    
    save_metadata(
        output_dir=output_dir,
        config=config,
        checkpoint=checkpoint,
    )


def export_to_onnx(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
    config: DictConfig,
    token_dict: Optional[dict] = None,
):
    """Export model to ONNX format."""
    print(f"Exporting to ONNX: {output_path}")
    
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("ONNX not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])
        import onnx
        import onnxruntime as ort
    
    # Export model
    torch.onnx.export(
        model,
        sample_input,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size", 2: "sequence_length"}
        },
        verbose=False,
    )
    
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    ort_session = ort.InferenceSession(
        str(output_path),
        providers=['CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    )
    
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    
    print(f"✓ ONNX model exported successfully")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output shape: {ort_outs[0].shape}")
    
    onnx_config = {
        "input_shape": [1, 3, config.data.image_size, config.data.image_size],
        "output_shape": [1, config.model.output.sequence_length, config.model.output.num_classes],
        "opset_version": 13,
    }
    
    import json
    with open(output_path.parent / "onnx_config.json", "w") as f:
        json.dump(onnx_config, f, indent=2)


def export_to_torchscript(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
):
    """Export model to TorchScript."""
    print(f"Exporting to TorchScript: {output_path}")
    
    traced_model = torch.jit.trace(model, sample_input)
    
    with torch.no_grad():
        original_output = model(sample_input)
        traced_output = traced_model(sample_input)
    
    if torch.allclose(original_output, traced_output, rtol=1e-3):
        print("✓ TorchScript model exported successfully")
    else:
        print("⚠ TorchScript model outputs differ slightly from original")
    
    traced_model.save(str(output_path))
    
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.save(str(output_path.with_name("model_scripted.pt")))
        print("✓ Scripted TorchScript model exported")
    except Exception as e:
        print(f"⚠ Could not export scripted model: {e}")


def save_metadata(
    output_dir: Path,
    config: DictConfig,
    checkpoint: dict,
):
    """Save metadata for deployment."""
    print("Saving metadata...")
    
    config_path = output_dir / "config.yaml"
    OmegaConf.save(config=config, f=config_path)
    
    if "token_dict" in checkpoint:
        import pickle
        token_dict_path = output_dir / "token_dictionary.pkl"
        with open(token_dict_path, "wb") as f:
            pickle.dump(checkpoint["token_dict"], f)
    
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "onnxruntime>=1.15.0" if (output_dir / "model.onnx").exists() else None,
    ]
    
    requirements = [r for r in requirements if r is not None]
    
    with open(output_dir / "requirements.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    deployment_info = {
        "model_formats": [f.stem for f in output_dir.glob("model.*")],
        "config_file": "config.yaml",
        "token_dict_file": "token_dictionary.pkl" if (output_dir / "token_dictionary.pkl").exists() else None,
        "input_shape": [1, 3, config.data.image_size, config.data.image_size],
        "output_classes": config.model.output.num_classes,
        "sequence_length": config.model.output.sequence_length,
        "timestamp": torch.datetime.now().isoformat(),
    }
    
    import json
    with open(output_dir / "deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print("✓ Metadata saved")
    print(f"  Output directory: {output_dir}")
    print(f"  Available formats: {deployment_info['model_formats']}")


if __name__ == "__main__":
    export_model()