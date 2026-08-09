"""Transparent notebook-equivalent training engine."""
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sleep_staging.evaluation.evaluator import evaluate
from .checkpointing import save_checkpoint
from .early_stopping import EarlyStopping


class Trainer:
    """Train with AdamW, validation-accuracy scheduling, and early stopping."""
    def __init__(self, model, optimizer, criterion, device, config, checkpoint_path):
        self.model=model.to(device); self.optimizer=optimizer; self.criterion=criterion; self.device=device; self.config=config
        tc=config["training"]; self.scheduler=ReduceLROnPlateau(optimizer,mode="max",factor=tc["scheduler_factor"],patience=tc["scheduler_patience"])
        self.stopper=EarlyStopping(tc["patience"]); self.checkpoint_path=Path(checkpoint_path)

    def train_epoch(self, loader):
        """Run one optimization epoch and return loss and sample-level accuracy."""
        self.model.train(); total=0.0; truth=[]; predictions=[]
        for x,y,_ in loader:
            x=x.to(self.device); y=y.to(self.device); self.model.reset_state(len(x),self.device); self.optimizer.zero_grad()
            logits=self.model(x,sequence_start=True)["logits"]; loss=self.criterion(logits,y); loss.backward(); self.optimizer.step()
            total += loss.detach().item()*len(x); truth.extend(y.cpu().numpy()); predictions.extend(logits.detach().argmax(1).cpu().numpy())
        return total/len(loader.dataset), accuracy_score(truth,predictions)

    def fit(self, train_loader, validation_loader):
        """Fit through configured epochs and return metric history."""
        history=[]; tc=self.config["training"]; dc=self.config["dataset"]
        for epoch in range(1,tc["epochs"]+1):
            train_loss,train_acc=self.train_epoch(train_loader)
            metrics=evaluate(self.model,validation_loader,self.device,self.criterion,dc["num_classes"],dc["class_names"])
            val_acc=metrics["overall"]["accuracy"]; self.scheduler.step(val_acc)
            history.append({"epoch":epoch,"train_loss":train_loss,"train_acc":train_acc,"validation":metrics})
            improved,stop=self.stopper.update(val_acc)
            if improved: save_checkpoint(self.checkpoint_path,self.model,self.optimizer,epoch,self.stopper.best,self.config)
            if stop: break
        return history
