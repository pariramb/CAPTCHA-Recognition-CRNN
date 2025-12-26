import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
import git
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger


def setup_mlflow(config: DictConfig, experiment_name: str = "captcha-recognizer"):
    """Setup MLflow logging."""
    if config.logging.enabled:
        mlflow.set_tracking_uri(config.logging.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        try:
            repo = git.Repo(search_parent_directories=True)
            git_commit = repo.head.object.hexsha
        except:
            git_commit = "unknown"
        
        mlflow.log_params(OmegaConf.to_container(config))
        mlflow.log_param("git_commit", git_commit)
        
        return MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=config.logging.mlflow_tracking_uri,
        )
    return None


def save_plots(trainer: pl.Trainer, output_dir: Path):
    """Save training plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trainer.logged_metrics["train_loss"].cpu().numpy(), label="Train Loss")
    ax.plot(trainer.logged_metrics["val_loss"].cpu().numpy(), label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    fig.savefig(output_dir / "loss_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trainer.logged_metrics["train_char_accuracy"].cpu().numpy(), label="Train Char Accuracy")
    ax.plot(trainer.logged_metrics["val_char_accuracy"].cpu().numpy(), label="Val Char Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Character Accuracy")
    ax.legend()
    ax.grid(True)
    fig.savefig(output_dir / "accuracy_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)