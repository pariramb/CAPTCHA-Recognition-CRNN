from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence

import fire
import lightning as L
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from captcha_rec.data.datamodule import CaptchaDataModule
from captcha_rec.data.download_data import download_data
from captcha_rec.export.export_onnx import export_onnx
from captcha_rec.export.export_trt import export_tensorrt
from captcha_rec.infer.mlflow_register import register_model
from captcha_rec.infer.predict import run_onnx_infer
from captcha_rec.models.lightning_module import LACCModule
from captcha_rec.utils.git import get_git_commit_id
from captcha_rec.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compose_config(config_name: str, overrides: Sequence[str]) -> Any:
    root = _repo_root()
    config_dir = root / "configs"

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def _dvc_pull(paths: Sequence[str]) -> None:
    cmd = ["dvc", "pull", *list(paths)]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("DVC pull failed")
        raise RuntimeError("dvc pull failed (see logs above).")


def _save_plots(series: Dict[str, list[float]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    if series.get("train_loss") and series.get("val_loss"):
        max_len = max(len(series["train_loss"]), len(series["val_loss"]))
        x = list(range(1, max_len + 1))
        plt.figure()
        plt.plot(
            x[: len(series["train_loss"])],
            series["train_loss"],
            label="train_loss",
        )
        plt.plot(
            x[: len(series["val_loss"])],
            series["val_loss"],
            label="val_loss",
        )
        plt.title("Loss curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curves.png")
        plt.close()

    if series.get("val_char_acc"):
        x = list(range(1, len(series["val_char_acc"]) + 1))
        plt.figure()
        plt.plot(x, series["val_char_acc"], label="val_char_acc")
        plt.title("Validation char accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "val_char_acc.png")
        plt.close()

    if series.get("val_seq_acc"):
        x = list(range(1, len(series["val_seq_acc"]) + 1))
        plt.figure()
        plt.plot(x, series["val_seq_acc"], label="val_seq_acc")
        plt.title("Validation sequence accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "val_seq_acc.png")
        plt.close()


def _to_plain_dict(cfg: Any) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


class Commands:

    def download_data(self, *overrides: str) -> None:
        cfg = _compose_config("train", overrides)
        setup_logging(cfg.logging.level, Path(cfg.logging.log_file))

        data_root = Path(cfg.data.root)
        download_data(data_root, cfg.data.dvc_storage)

    def train(self, *overrides: str) -> None:
        cfg = _compose_config("train", overrides)
        setup_logging(cfg.logging.level, Path(cfg.logging.log_file))
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        L.seed_everything(int(cfg.seed), workers=True)

        download_data(Path(cfg.data.root), cfg.data.dvc_storage)

        dm = CaptchaDataModule(
            data_root=str(cfg.data.root),
            image_size=int(cfg.data.image_size),
            batch_size=int(cfg.data.batch_size),
            num_workers=int(cfg.data.num_workers),
            max_len=int(cfg.model.max_len),
            pin_memory=bool(cfg.data.pin_memory),
        )
        dm.setup()
        lit = LACCModule(
            vocab_size=dm.vocab_size,
            max_len=int(cfg.model.max_len),
            lr=float(cfg.model.lr),
            weight_decay=float(cfg.model.weight_decay),
            pad_id=dm.pad_id,
            optimizer_name=str(cfg.model.optimizer),
        )

        mlflow_logger = MLFlowLogger(
            tracking_uri=str(cfg.logger.mlflow.tracking_uri),
            experiment_name=str(cfg.logger.mlflow.experiment_name),
            run_name=str(cfg.logger.mlflow.run_name),
        )

        plain = _to_plain_dict(cfg)
        plain["git_commit_id"] = get_git_commit_id()
        mlflow_logger.log_hyperparams(plain)

        ckpt_dir = Path(cfg.paths.checkpoints_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = L.Trainer(
            max_epochs=int(cfg.trainer.max_epochs),
            accelerator=str(cfg.trainer.accelerator),
            devices=cfg.trainer.devices,
            precision=str(cfg.trainer.precision),
            log_every_n_steps=int(cfg.trainer.log_every_n_steps),
            callbacks=[checkpoint_cb, lr_monitor],
            logger=mlflow_logger,
            enable_checkpointing=True,
        )

        logger.info("Starting training...")
        trainer.fit(lit, datamodule=dm)

        plots_dir = Path(cfg.paths.plots_dir)
        series = lit.export_plot_series()
        _save_plots(series, plots_dir)
        logger.info("Plots saved to: %s", plots_dir)

        best_ckpt = checkpoint_cb.best_model_path
        if best_ckpt:
            export_onnx(
                checkpoint_path=Path(best_ckpt),
                onnx_path=Path(cfg.export.onnx_path),
                image_size=int(cfg.data.image_size),
                vocab_size=dm.vocab_size,
                max_len=int(cfg.model.max_len),
            )

        if best_ckpt and bool(cfg.export.auto_export_trt):
            export_tensorrt(
                onnx_path=Path(cfg.export.onnx_path),
                engine_path=Path(cfg.export.trt_engine_path),
                fp16=bool(cfg.export.trt_fp16),
                workspace_mb=int(cfg.export.trt_workspace_mb),
            )

        logger.info("Training finished. Best checkpoint: %s", best_ckpt)

    def export_onnx(self, *overrides: str) -> None:
        cfg = _compose_config("train", overrides)
        setup_logging(cfg.logging.level, Path(cfg.logging.log_file))

        ckpt_dir = Path(cfg.paths.checkpoints_dir)
        ckpt_path = ckpt_dir / "best.ckpt"
        if not ckpt_path.exists():
            candidates = sorted(
                ckpt_dir.glob("*.ckpt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            ckpt_path = candidates[0]

        dm = CaptchaDataModule(
            data_root=str(cfg.data.root),
            image_size=int(cfg.data.image_size),
            batch_size=int(cfg.data.batch_size),
            num_workers=int(cfg.data.num_workers),
            max_len=int(cfg.model.max_len),
            pin_memory=bool(cfg.data.pin_memory),
        )
        dm.setup()

        export_onnx(
            checkpoint_path=ckpt_path,
            onnx_path=Path(cfg.export.onnx_path),
            image_size=int(cfg.data.image_size),
            vocab_size=dm.vocab_size,
            max_len=int(cfg.model.max_len),
        )

    def export_trt(self, *overrides: str) -> None:
        cfg = _compose_config("train", overrides)
        setup_logging(cfg.logging.level, Path(cfg.logging.log_file))

        export_tensorrt(
            onnx_path=Path(cfg.export.onnx_path),
            engine_path=Path(cfg.export.trt_engine_path),
            fp16=bool(cfg.export.trt_fp16),
            workspace_mb=int(cfg.export.trt_workspace_mb),
        )

    def infer(self, *overrides: str) -> None:
        cfg = _compose_config("infer", overrides)
        setup_logging(cfg.logging.level, Path(cfg.logging.log_file))
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        run_onnx_infer(
            onnx_path=Path(cfg.infer.onnx_path),
            inputs=list(cfg.infer.inputs),
            output_jsonl=Path(cfg.infer.output),
            image_size=int(cfg.data.image_size),
        )

    def register_model_mlflow(self, *overrides: str) -> None:
        cfg = _compose_config("infer", overrides)
        setup_logging(cfg.logging.level, Path(cfg.logging.log_file))

        register_model(cfg.infer.onnx_path)


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
