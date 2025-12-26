import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from omegaconf import DictConfig


class LACC(pl.LightningModule):
    """LAbel Combination Classifier for CAPTCHA recognition."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.encoder = torchvision.models.efficientnet_v2_m().features
        
        self.converter = nn.Parameter(torch.ones(64, config.model.output.num_classes))
        
        self.silu = nn.SiLU()
        self.linear1 = nn.Linear(1280, config.model.hidden_size)
        self.dropout = nn.Dropout(config.model.dropout)
        self.linear2 = nn.Linear(config.model.hidden_size, 64)
        self.linear3 = nn.Linear(64, config.model.output.sequence_length)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=2)
        features = torch.matmul(features, self.converter)
        
        y = features.transpose(-1, -2)
        y = self.linear1(y)
        y = self.silu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.silu(y)
        y = self.linear3(y)
        
        return y
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, labels, _ = batch
        logits = self(images)
        
        batch_size, seq_len, num_classes = logits.shape
        logits = logits.permute(0, 2, 1)  # [batch, num_classes, seq_len]
        
        loss = self.criterion(logits, labels)
        
        predictions = torch.argmax(logits, dim=1)
        char_accuracy = (predictions == labels).float().mean()
        seq_accuracy = (predictions == labels).all(dim=1).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_char_accuracy", char_accuracy, prog_bar=True)
        self.log("train_seq_accuracy", seq_accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels, _ = batch
        logits = self(images)
        
        batch_size, seq_len, num_classes = logits.shape
        logits = logits.permute(0, 2, 1)
        
        loss = self.criterion(logits, labels)
        
        predictions = torch.argmax(logits, dim=1)
        char_accuracy = (predictions == labels).float().mean()
        seq_accuracy = (predictions == labels).all(dim=1).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_char_accuracy", char_accuracy, prog_bar=True)
        self.log("val_seq_accuracy", seq_accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        if self.config.training.optimizer == "Lion":
            from lion_pytorch import Lion
            optimizer = Lion(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.training.learning_rate,
            )
        
        if self.config.training.scheduler.enabled:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.scheduler.T_max,
                eta_min=self.config.training.scheduler.eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return optimizer