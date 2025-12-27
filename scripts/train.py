#!/usr/bin/env python3
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torchvision
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloaders
from src.model import create_model
from src.train import Trainer, load_checkpoint, save_checkpoint
from src.utils import set_random_seed, setup_matplotlib, clean_memory, plot_loss
from src.inference import Inference
from src.config import setup_config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg = setup_config(cfg)
    
    print(OmegaConf.to_yaml(cfg))
    
    clean_memory()
    set_random_seed(cfg.seed)
    
    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    device = torch.device(cfg.device if use_cuda else "cpu")
    print(f"Device: {device}")
    
    fontprop = None
    if cfg.inference.visualization.get('font_path'):
        fontprop = setup_matplotlib(cfg.inference.visualization.font_path)
    
    print(f"Torch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")

    print("\nCreating dataloaders...")
    train_dataloader, test_dataloader = create_dataloaders(cfg, mode="train")

    print("\nCreating model...")
    model = create_model(cfg, device)
    
    if cfg.optimizer.name.lower() == "lion":
        from lion_pytorch import Lion
        optimizer = Lion(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas)
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas)
        )
    
    if cfg.optimizer.loss.name == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg.optimizer.loss.get('label_smoothing', 0.0)
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    start_epoch = cfg.training.start_epoch
    if cfg.training.checkpoint.get('save_best') and os.path.exists(cfg.checkpoint_path):
        checkpoint_file = os.path.join(cfg.checkpoint_path, "best_model.pth")
        if os.path.exists(checkpoint_file):
            start_epoch = load_checkpoint(model, optimizer, checkpoint_file, device)
            print(f"Loaded checkpoint from epoch {start_epoch}")
    
    print(f"\nStarting training from epoch {start_epoch}...")
    trainer = Trainer(model, optimizer, criterion, device, cfg)
    
    train_losses, test_losses, accuracies = trainer.train(
        train_dataloader,
        test_dataloader,
        epochs=cfg.training.epochs,
        start_epoch=start_epoch
    )

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plot_loss(train_losses, test_losses, start_epoch)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(start_epoch, start_epoch + len(accuracies)), accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "training_plots.png"))
    plt.show()

    print("\nTesting inference on samples...")
    inference = Inference(model, device, cfg)
    results = inference.test_samples(test_dataloader, num_samples=10)
    
    correct = sum(1 for r in results if r['correct'])
    print(f"\nSample Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")

    print("\nSaving model...")
    save_checkpoint(
        model,
        optimizer,
        start_epoch + cfg.training.epochs - 1,
        os.path.join(cfg.save_dir, "final_model.pth"),
        cfg
    )


if __name__ == "__main__":
    main()