"""Validation-accuracy early stopping."""

class EarlyStopping:
    """Track strict improvements, matching ``val_acc > best`` in the notebook."""
    def __init__(self, patience=30): self.patience=patience; self.best=0.0; self.bad_epochs=0
    def update(self, value):
        improved = value > self.best
        if improved: self.best=float(value); self.bad_epochs=0
        else: self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
