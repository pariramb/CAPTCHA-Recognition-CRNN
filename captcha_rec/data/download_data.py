from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)


def download_from_kaggle(dvc_storage: str):
    try:
        api = KaggleApi()
        api.authenticate()
        datasets = [
            "jassoncarvalho/comprasnet-captchas",
            "parsasam/captcha-dataset",
            "aadhavvignesh/captcha-images",
            "fournierp/captcha-version-2-images",
            "lapl04/koreanfont",
            "akashguna/large-captcha-dataset",
        ]

        for dataset in datasets:
            logger.info(f"Installing {dataset} dataset...")
            api.dataset_download_files(dataset, path=dvc_storage, unzip=True)
    except Exception as e:
        logger.error(f"Error: {e}")

    logger.info(f"Datasets are installed in: {dvc_storage}")


def dvc_pull(target_dir: Path) -> bool:
    try:
        logger.info(f"Make dvc pull for {target_dir}...")

        result = subprocess.run(
            ["dvc", "pull", str(target_dir)], capture_output=True, text=True
        )

        if result.returncode == 0:
            logger.info("dvc pull success")
            return True
        else:
            logger.warning(f"dvc pull failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Exception while dvc pull: {e}")
        return False


def setup_local_storage(dvc_storage: str) -> bool:
    try:
        storage = Path(dvc_storage)
        storage.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["dvc", "remote", "list"], capture_output=True, text=True
        )

        if "localstorage" not in result.stdout:
            logger.info(f"Configuring local storage: {dvc_storage}")
            subprocess.run(
                ["dvc", "remote", "add", "-d", "localstorage", dvc_storage],
                check=True,
            )
            logger.info("Local storage configured")

        return True
    except Exception as e:
        logger.error(f"Storage configuration error: {e}")
        return False


def check_data_in_storage(dvc_storage: str) -> bool:
    """Check if data exists in local DVC storage"""
    storage = Path(dvc_storage)
    if not storage.exists():
        return False

    for item in storage.rglob("*"):
        if item.is_file() and item.stat().st_size > 0:
            return True

    return False


def download_data(data_root: Path, dvc_storage: str) -> None:
    """
    Main function for downloading data.
    Checks data availability, downloads if needed and integrates with DVC.
    """
    data_root.mkdir(parents=True, exist_ok=True)

    if data_root.exists():
        files = [
            f
            for f in data_root.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        ]
        if len(files) > 0:
            logger.info(f"Data already exists in {data_root}")
            return

    logger.info("Checking local DVC storage...")
    if check_data_in_storage(dvc_storage):
        logger.info("Data found in local DVC storage")
    else:
        logger.info("Data not found. Starting download...")
        download_from_kaggle(dvc_storage)

    setup_local_storage(dvc_storage)
    if dvc_pull(data_root):
        logger.info(f"Data restored from DVC storage to {data_root}")
        return
    else:
        logger.warning("Failed to execute dvc pull, downloading again...")

    files_after = list(data_root.rglob("*"))
    if len(files_after) == 0:
        logger.error("Failed to download data from any source")
        return

    logger.info(f"Data downloaded ({len(files_after)} files)")
