from __future__ import annotations

import logging
import os
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


def dvc_pull() -> bool:
    try:
        logger.info("Make dvc pull...")

        result = subprocess.run(
            ["dvc", "pull"],
            capture_output=True,
            text=True,
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


def dvc_push() -> bool:
    try:
        logger.info("Make dvc push...")

        result = subprocess.run(["dvc", "pu"], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("dvc pull success")
            return True
        else:
            logger.warning(f"dvc pull failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Exception while dvc pull: {e}")
        return False


def setup_local_storage(dvc_storage: str, data_root: str) -> bool:
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
        subprocess.run(
            ["dvc", "add", data_root],
            check=True,
        )
        subprocess.run(
            ["dvc", "push"],
            check=True,
        )
        logger.info("Local storage configured")

        return True
    except Exception as e:
        logger.error(f"Storage configuration error: {e}")
        return False


def check_dvc_hash_via_cli(dvc_file="data.dvc"):
    try:
        result = subprocess.run(
            ["dvc", "status", dvc_file],
            capture_output=True,
            text=True,
            check=True,
        )

        if "changed" in result.stdout.lower():
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"DVC command execution error: {e}")
        return False
    except FileNotFoundError:
        print("DVC is not installed or not found in PATH")
        return False


def check_data_in_storage(dvc_storage: str) -> bool:
    storage = Path(dvc_storage)
    if not storage.exists():
        return False

    for item in storage.rglob("*"):
        if item.is_file() and item.stat().st_size > 0:
            return True

    return False


def download_data(data_root_path: str, dvc_storage_path: str) -> None:
    data_root = Path(f"{os.getcwd()}/{data_root_path}")
    data_root.mkdir(parents=True, exist_ok=True)

    ls = data_root.rglob("*")
    files = [f for f in ls if f.is_file() and not f.name.startswith(".")]

    if len(files) > 0:
        logger.info(f"Data already exists in {data_root}")
    else:
        logger.info(f"Data not found in {data_root}. Starting download...")
        download_from_kaggle(data_root)

        files_after = list(data_root.rglob("*"))
        if len(files_after) == 0:
            logger.error("Failed to download data from any source")
            return
        logger.info(f"Data downloaded ({len(files_after)} files)")
    if not check_data_in_storage(dvc_storage_path):
        logger.info(f"Setup storage: {dvc_storage_path}")
        setup_local_storage(dvc_storage_path, data_root_path)
    if check_dvc_hash_via_cli():
        logger.info("Data is correct")
    else:
        logger.info("Data hash was changed")
