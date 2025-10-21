from __future__ import annotations
import numpy as np
from pathlib import Path

def save_run(out_dir: Path, **arrays):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "run_data.npz", **arrays)
