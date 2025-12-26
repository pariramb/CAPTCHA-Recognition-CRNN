import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from omegaconf import DictConfig


class CaptchaDataset(Dataset):
    """Dataset for CAPTCHA images."""

    def __init__(
        self,
        dataset_paths: List[str],
        token_dict: dict,
        max_length: int = 10,
        transform: Optional[transforms.Compose] = None,
        ban_data: Optional[List[str]] = None,
    ):
        self.dataset_paths = dataset_paths
        self.token_dict = token_dict
        self.reverse_token_dict = {v: k for k, v in token_dict.items()}
        self.max_length = max_length
        self.transform = transform
        self.ban_data = ban_data or []
        
        self.file_list = self._load_file_list()
        
    def _load_file_list(self) -> List[str]:
        """Load all image files from dataset paths."""
        file_list = []
        for dataset_path in self.dataset_paths:
            if os.path.exists(dataset_path):
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    file_list.extend(glob.glob(os.path.join(dataset_path, ext)))
        
        file_list = [f for f in file_list if f not in self.ban_data]
        return file_list
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filename = self.file_list[idx]
        
        label_str = Path(filename).stem
        label_indices = []
        
        for char in label_str[:self.max_length]:
            if char in self.reverse_token_dict:
                label_indices.append(self.reverse_token_dict[char])
            else:
                continue
        
        if len(label_indices) < self.max_length:
            pad_idx = self.reverse_token_dict["<pad>"]
            label_indices += [pad_idx] * (self.max_length - len(label_indices))
        
        image = cv2.imread(filename)
        if image is None:
            raise ValueError(f"Failed to load image: {filename}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        return image, label_tensor, label_tensor


class CaptchaDataModule:
    """PyTorch Lightning DataModule for CAPTCHA."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.token_dict = self._create_token_dict()
        
    def _create_token_dict(self) -> dict:
        """Create token dictionary from config."""
        all_chars = (
            self.config.data.special_chars +
            self.config.data.numerals +
            self.config.data.uppercase +
            self.config.data.lowercase
        )
        return {i: char for i, char in enumerate(all_chars)}
    
    def setup(self, stage: str = None):
        """Setup datasets for training/validation."""
        transform = self._get_transforms()
        
        full_dataset = CaptchaDataset(
            dataset_paths=self.config.data.dataset_paths,
            token_dict=self.token_dict,
            max_length=self.config.data.max_length,
            transform=transform,
            ban_data=self.config.data.ban_data,
        )
        
        train_size = int(self.config.data.train_test_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image transformations."""
        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((self.config.data.image_size, self.config.data.image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        
        if self.config.data.augmentation.enabled and hasattr(self, "train_dataset"):
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            aug_list = []
            for aug in self.config.data.augmentation.transforms:
                if aug.type == "RandomRotation":
                    aug_list.append(A.Rotate(limit=aug.degrees))
                elif aug.type == "RandomBrightnessContrast":
                    aug_list.append(A.RandomBrightnessContrast(
                        brightness_limit=aug.brightness_limit,
                        contrast_limit=aug.contrast_limit
                    ))
                elif aug.type == "GaussNoise":
                    aug_list.append(A.GaussNoise(var_limit=aug.var_limit))
            
            aug_list.append(A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            aug_list.append(ToTensorV2())
            
            return A.Compose(aug_list)
        
        return transforms.Compose(transform_list)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )