#!/usr/bin/env python3
"""Run all configured grouped cross-validation folds."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True); parser.add_argument("--data-dir",required=True); parser.add_argument("--output-dir",default="outputs/cross_validation"); parser.add_argument("--device"); args=parser.parse_args()
    import yaml
    with Path(args.config).open() as stream: folds=yaml.safe_load(stream)["cross_validation"]["folds"]
    train_script=Path(__file__).with_name("train.py")
    for fold in range(1,folds+1):
        command=[sys.executable,str(train_script),"--config",args.config,"--data-dir",args.data_dir,"--output-dir",args.output_dir,"--fold",str(fold)]
        if args.device: command.extend(["--device",args.device])
        subprocess.run(command,check=True)


if __name__ == "__main__": main()
