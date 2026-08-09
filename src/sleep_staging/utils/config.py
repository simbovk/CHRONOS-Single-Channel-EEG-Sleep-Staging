"""YAML configuration loading."""
from pathlib import Path
import yaml


def load_config(path):
    """Load and minimally validate an experiment YAML file."""
    with Path(path).open(encoding="utf-8") as stream: config=yaml.safe_load(stream)
    for section in ("dataset","model","training","cross_validation"):
        if section not in config: raise ValueError(f"missing config section: {section}")
    return config
