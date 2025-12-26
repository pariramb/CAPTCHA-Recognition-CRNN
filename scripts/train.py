#!/usr/bin/env python3
"""Inference script for CAPTCHA recognizer."""

import argparse
from pathlib import Path
from typing import List, Optional
import hydra
from omegaconf import DictConfig
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from src.models.lacc import LACC
from src.data.dataset import CaptchaDataset


class CAPTCHAPredictor:
    """CAPTCHA prediction class."""
    
    def __init__(self, config_path: str = "../configs/inference.yaml"):
        with hydra.initialize(version_base=None, config_path="../configs"):
            self.config = hydra.compose(config_name="inference")
        
        checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
        self.model = LACC.load_from_checkpoint(self.config.model_path, strict=False)
        self.model.eval()
        self.model.to(self.config.device)
        
        with open(self.config.token_dict_path, "rb") as f:
            import pickle
            self.token_dict = pickle.load(f)
        self.reverse_token_dict = {v: k for k, v in self.token_dict.items()}
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.config.device)
    
    def decode_prediction(self, logits: torch.Tensor) -> str:
        """Decode model output to string."""
        predictions = torch.argmax(logits, dim=1)[0]
        prediction_str = ""
        
        for idx in predictions.cpu().numpy():
            if idx in self.token_dict and self.token_dict[idx] != "<pad>":
                prediction_str += self.token_dict[idx]
        
        return prediction_str
    
    def predict(self, image_path: str) -> str:
        """Predict CAPTCHA from image."""
        image_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
        
        prediction = self.decode_prediction(logits)
        return prediction
    
    def predict_batch(self, image_paths: List[str]) -> List[str]:
        """Predict batch of images."""
        predictions = []
        
        for image_path in image_paths:
            try:
                prediction = self.predict(image_path)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                predictions.append("")
        
        return predictions


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="CAPTCHA Recognizer Inference")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--input_dir", type=str, help="Path to input directory")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="Output CSV file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    predictor = CAPTCHAPredictor()
    
    if args.image:
        prediction = predictor.predict(args.image)
        print(f"Image: {args.image}")
        print(f"Prediction: {prediction}")
    
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        image_paths = [str(f) for f in image_files]
        
        predictions = predictor.predict_batch(image_paths)
        
        df = pd.DataFrame({
            "image": image_paths,
            "prediction": predictions,
        })
        df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
        print(f"Processed {len(predictions)} images")
    
    else:
        print("Please provide either --image or --input_dir argument")


if __name__ == "__main__":
    main()