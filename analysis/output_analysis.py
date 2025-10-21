# Shitty_SNN/analysis/output_analysis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import csv

def analyze_output(counts_out: np.ndarray, T_bins: int, bin_ms: int, path_dir: Path):
    n_out = counts_out.shape[0]
    total = int(counts_out.sum())
    winner = int(np.argmax(counts_out)) if n_out > 0 else -1
    duration_s = T_bins * (bin_ms / 1000.0)
    rate_hz = total / duration_s if duration_s > 0 else 0.0

    p = counts_out.astype(np.float64)
    if p.sum() > 0:
        p /= p.sum()
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    else:
        entropy = 0.0

    topk = np.argsort(counts_out)[-5:][::-1]

    print("\n=== OUTPUT ANALYSIS ===")
    print(f"Winner index: {winner}  (count={int(counts_out[winner]) if winner>=0 else 0})")
    print(f"Total spikes: {total}")
    print(f"Avg firing rate: {rate_hz:.4f} Hz")
    print(f"Entropy (bits): {entropy:.4f}")
    print(f"Top-5 units: {topk.tolist()} with counts {counts_out[topk].astype(int).tolist()}")

    path_dir.mkdir(parents=True, exist_ok=True)
    (path_dir / "output_analysis.txt").write_text(
        "\n".join([
            f"n_out: {n_out}",
            f"winner_index: {winner}",
            f"winner_count: {int(counts_out[winner]) if winner>=0 else 0}",
            f"total_output_spikes: {total}",
            f"avg_rate_hz: {rate_hz:.6f}",
            f"entropy_bits: {entropy:.6f}",
            f"top5_indices: {topk.tolist()}",
            f"top5_counts: {counts_out[topk].astype(int).tolist()}",
        ]),
        encoding="utf-8"
    )

    with open(path_dir / "output_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit", "spike_count"])
        for i, c in enumerate(counts_out):
            w.writerow([i, int(c)])

    return {
        "n_out": n_out,
        "winner_index": winner,
        "winner_count": int(counts_out[winner]) if winner >= 0 else 0,
        "total_output_spikes": total,
        "avg_rate_hz": rate_hz,
        "entropy_bits": entropy,
        "top5_indices": topk.tolist(),
        "top5_counts": counts_out[topk].astype(int).tolist(),
    }
