"""Small JSON result writer."""
import json
from pathlib import Path


def write_json(path, value):
    """Write JSON, creating parent directories."""
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(value,indent=2),encoding="utf-8")
