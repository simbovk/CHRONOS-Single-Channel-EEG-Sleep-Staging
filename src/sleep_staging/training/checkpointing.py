"""Checkpoint persistence."""
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, best_metric, config):
    """Save enough state for resuming and evaluation."""
    target=Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_val_acc": best_metric,
                "config": config}, target)


def load_checkpoint(path, model, device="cpu", optimizer=None):
    """Load a notebook-style state dict or structured checkpoint."""
    checkpoint=torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    if optimizer is not None and "optimizer_state_dict" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
