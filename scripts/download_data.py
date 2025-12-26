#!/usr/bin/env python3
"""Script to download CAPTCHA datasets."""

import kagglehub
from pathlib import Path
import dvc.api
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def download_data(config: DictConfig):
    """Download CAPTCHA datasets using kagglehub."""
    
    print("Downloading CAPTCHA datasets...")
    
    datasets = [
        ("fournierp/captcha-version-2-images", "captcha-version-2-images"),
        ("aadhavvignesh/captcha-images", "captcha-images"),
        ("parsasam/captcha-dataset", "captcha-dataset"),
        ("akashguna/large-captcha-dataset", "large-captcha-dataset"),
        ("jassoncarvalho/comprasnet-captchas", "comprasnet-captchas"),
    ]
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for dataset_slug, dataset_name in datasets:
        try:
            print(f"Downloading {dataset_name}...")
            path = kagglehub.dataset_download(dataset_slug)
            print(f"Downloaded to: {path}")
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")
    
    print("Data download complete!")
    
    if not Path(".dvc").exists():
        print("Initializing DVC...")
        import subprocess
        subprocess.run(["dvc", "init"], check=True)
    
    print("Adding data to DVC...")
    import subprocess
    subprocess.run(["dvc", "add", "data/"], check=True)
    subprocess.run(["git", "add", "data.dvc", ".gitignore"], check=True)


if __name__ == "__main__":
    download_data()