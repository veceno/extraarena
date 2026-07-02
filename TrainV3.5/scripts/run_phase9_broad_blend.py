#!/usr/bin/env python3
"""Run Extra-LR V5 Phase 9 broad-opponent blend training."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrainV3.5" / "python"))

from train_v3.phase9_broad_runner import run_phase9_broad_training


if __name__ == "__main__":
    raise SystemExit(0 if run_phase9_broad_training(ROOT) else 1)
