#!/usr/bin/env python3
"""Evaluate a checkpoint on a configured fold."""
import argparse
from sleep_staging.data.loaders import load_arrays, make_loader
from sleep_staging.data.splits import generate_folds
from sleep_staging.evaluation.evaluator import evaluate
from sleep_staging.models.sleep_staging_model import SleepStagingModel
from sleep_staging.training.checkpointing import load_checkpoint
from sleep_staging.utils.config import load_config
from sleep_staging.utils.device import get_device
from sleep_staging.utils.logging import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); parser.add_argument("--data-dir",required=True); parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--fold",type=int,default=1); parser.add_argument("--output",default="outputs/evaluation.json"); parser.add_argument("--device"); args=parser.parse_args()
    config=load_config(args.config); device=get_device(args.device); X,y,groups=load_arrays(args.data_dir,config["dataset"])
    _,indices=list(generate_folds(y,groups,config["cross_validation"]["folds"],config["cross_validation"]["shuffle"],config["cross_validation"]["seed"]))[args.fold-1]
    tc=config["training"]; loader=make_loader(X,y,groups,indices=indices,channel_indices=config["dataset"]["channel_indices"],batch_size=tc["batch_size"],num_workers=tc["num_workers"])
    model=SleepStagingModel.from_config(config["model"]).to(device); load_checkpoint(args.checkpoint,model,device)
    write_json(args.output,evaluate(model,loader,device,num_classes=config["dataset"]["num_classes"],class_names=config["dataset"]["class_names"]))


if __name__ == "__main__": main()
