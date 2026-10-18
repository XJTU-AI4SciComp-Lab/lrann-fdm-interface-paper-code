"""Run a small square-interface problem and save its fields and diagnostics."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrann_interface.runner import run

if __name__ == "__main__":
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    run(config)
