#!/usr/bin/env python3
"""Train one configured train/validation fold."""
import argparse
from pathlib import Path
import numpy as np
import torch
from sleep_staging.data.augmentation import EEGAugment
from sleep_staging.data.loaders import load_arrays, make_loader
from sleep_staging.data.preprocessing import validate_arrays
from sleep_staging.data.splits import generate_folds
from sleep_staging.models.sleep_staging_model import SleepStagingModel
from sleep_staging.training.losses import compute_class_weights, make_loss
from sleep_staging.training.trainer import Trainer
from sleep_staging.utils.config import load_config
from sleep_staging.utils.device import get_device
from sleep_staging.utils.logging import write_json
from sleep_staging.utils.seed import set_seed


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); parser.add_argument("--data-dir",required=True)
    parser.add_argument("--output-dir",default="outputs/train"); parser.add_argument("--fold",type=int,default=1); parser.add_argument("--device")
    return parser.parse_args()


def main():
    args=parse_args(); config=load_config(args.config); device=get_device(args.device); set_seed(config["cross_validation"]["seed"])
    X,y,groups=load_arrays(args.data_dir,config["dataset"]); validate_arrays(X,y,groups,config["dataset"]["num_classes"])
    folds=list(generate_folds(y,groups,config["cross_validation"]["folds"],config["cross_validation"]["shuffle"],config["cross_validation"]["seed"]))
    if not 1 <= args.fold <= len(folds): raise ValueError("fold is out of range")
    train_idx,val_idx=folds[args.fold-1]; tc=config["training"]; channels=config["dataset"]["channel_indices"]
    transform=EEGAugment.from_config(config["augmentation"]) if config["augmentation"].get("enabled") else None
    train=make_loader(X,y,groups,indices=train_idx,channel_indices=channels,batch_size=tc["batch_size"],shuffle=True,num_workers=tc["num_workers"],transform=transform)
    validation=make_loader(X,y,groups,indices=val_idx,channel_indices=channels,batch_size=tc["batch_size"],num_workers=tc["num_workers"])
    model=SleepStagingModel.from_config(config["model"]); weights=compute_class_weights(y[train_idx],config["dataset"]["num_classes"],device)
    criterion=make_loss(weights); optimizer=torch.optim.AdamW(model.parameters(),lr=tc["learning_rate"],weight_decay=tc["weight_decay"])
    output=Path(args.output_dir); trainer=Trainer(model,optimizer,criterion,device,config,output/f"best_model_fold{args.fold}.pth")
    write_json(output/f"history_fold{args.fold}.json",trainer.fit(train,validation))


if __name__ == "__main__": main()
