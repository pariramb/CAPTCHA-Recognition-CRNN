from __future__ import annotations

from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion

from captcha_rec.models.lacc import LACC


class LACCModule(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        lr: float,
        weight_decay: float,
        pad_id: int,
        optimizer_name: str = "lion",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = LACC(vocab_size=vocab_size, max_len=max_len)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_id = int(pad_id)

        self._epoch_train_losses: list[float] = []
        self._epoch_val_losses: list[float] = []
        self._epoch_val_char_acc: list[float] = []
        self._epoch_val_seq_acc: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _char_accuracy(pred: torch.Tensor, target: torch.Tensor):
        return (pred == target).float().mean()

    @staticmethod
    def _seq_accuracy(pred: torch.Tensor, target: torch.Tensor):
        return (pred == target).all(dim=1).float().mean()

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)

        log_probs = F.log_softmax(logits, dim=-2)
        loss = self.criterion(log_probs, y)
        pred = torch.argmax(log_probs, dim=-2)

        char_acc = self._char_accuracy(pred, y)
        seq_acc = self._seq_accuracy(pred, y)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}/char_acc",
            char_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}/seq_acc",
            seq_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        if "train/loss" in metrics:
            self._epoch_train_losses.append(
                float(metrics["train/loss"].detach().cpu().item())
            )

    def on_validation_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        if "val/loss" in metrics:
            self._epoch_val_losses.append(
                float(metrics["val/loss"].detach().cpu().item())
            )
        if "val/char_acc" in metrics:
            self._epoch_val_char_acc.append(
                float(metrics["val/char_acc"].detach().cpu().item())
            )
        if "val/seq_acc" in metrics:
            self._epoch_val_seq_acc.append(
                float(metrics["val/seq_acc"].detach().cpu().item())
            )

    def configure_optimizers(self):
        lr = float(self.hparams.lr)
        wd = float(self.hparams.weight_decay)
        name = str(self.hparams.optimizer_name).lower()

        if name == "lion":
            optimizer = Lion(self.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=wd,
            )

        return optimizer

    def export_plot_series(self) -> Dict[str, list[float]]:
        return {
            "train_loss": self._epoch_train_losses,
            "val_loss": self._epoch_val_losses,
            "val_char_acc": self._epoch_val_char_acc,
            "val_seq_acc": self._epoch_val_seq_acc,
        }
