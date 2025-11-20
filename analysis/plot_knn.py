#!/usr/bin/env python3
"""
Plot validation kNN@1 for diffusion and flow matching (from logs) and annotate
scalar checkpoints for the MDN/autoregressive heads.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "analysis" / "cache"


def load_cached_metrics():
    cache_path = CACHE_DIR / "knn_metrics.json"
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text())


def load_diffusion_knn():
    scalars_path = ROOT / "diffusion" / "scalars (2).json"
    steps, values = [], []
    if not scalars_path.exists():
        return steps, values
    data = json.loads(scalars_path.read_text())
    for rec in data.get("val/knn_top1", []):
        steps.append(rec["step"])
        values.append(rec["value"] * 100.0)
    return steps, values


def load_flow_knn():
    data = {}
    event_files = sorted((ROOT / "flow_matching" / "runs").glob("*/events.out.tfevents.*"))
    for event_path in event_files:
        acc = EventAccumulator(str(event_path))
        acc.Reload()
        scalars = acc.Tags().get("scalars", [])
        if "val/knn_top1" not in scalars:
            continue
        events = acc.Scalars("val/knn_top1")
        if not events:
            continue
        if events[0].step > 5_000_000 or len(events) < 5:
            continue
        for event in events:
            data[event.step] = event.value * 100.0
    steps = sorted(data)
    values = [data[s] for s in steps]
    return steps, values


def _find_baseline_ckpt():
    candidates = [
        ROOT / "I-JEPA" / "checkpoints" / "ijeja_best_knn.pt",
        ROOT.parent
        / "BionicEye"
        / "representation"
        / "current"
        / "checkpoints"
        / "ijeja_best_knn.pt",
        ROOT.parent
        / "BionicEye"
        / "representation"
        / "checkpoints"
        / "ijeja_best_knn.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_baseline_knn():
    ckpt_path = _find_baseline_ckpt()
    if ckpt_path is None:
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_knn = float(state.get("best_knn", 0.0) or 0.0) * 100.0
    samples_seen = float(state.get("samples_seen", 0.0) or 0.0)
    return samples_seen, best_knn


def main():
    diff_steps, diff_knn = load_diffusion_knn()
    flow_steps, flow_knn = load_flow_knn()
    baseline = load_baseline_knn()
    cache = load_cached_metrics()
    arn_knn = cache.get("autoregressive")
    mdn_knn = cache.get("mdn")

    plt.figure(figsize=(6.5, 4.2))
    if flow_steps:
        plt.plot(np.array(flow_steps) / 1e6, flow_knn, label="Flow val kNN@1")
    if diff_steps:
        plt.plot(np.array(diff_steps) / 1e6, diff_knn, label="Diffusion val kNN@1")

    if arn_knn is not None:
        plt.axhline(arn_knn, color="tab:green", linestyle="--", label=f"Autoregressive ({arn_knn:.2f}%)")
        plt.text(0.1, arn_knn + 0.3, f"RNADE {arn_knn:.2f}%", color="tab:green")

    if mdn_knn is not None:
        plt.axhline(mdn_knn, color="tab:orange", linestyle=":", label=f"MDN ({mdn_knn:.2f}%)")
        plt.text(0.1, mdn_knn + 0.3, f"MDN {mdn_knn:.2f}%", color="tab:orange")

    if baseline is not None:
        step, knn_pct = baseline
        plt.axhline(
            knn_pct,
            color="tab:purple",
            linestyle="-.",
            label=f"I-JEPA baseline ({knn_pct:.2f}%)",
        )
        plt.scatter([step / 1e6], [knn_pct], color="tab:purple", marker="x")
        plt.text(step / 1e6 + 0.05, knn_pct + 0.3, f"{knn_pct:.2f}%", color="tab:purple")

    plt.xlabel("Steps (millions)")
    plt.ylabel("kNN@1 (%)")
    plt.title("Validation kNN@1 across probabilistic heads")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = FIG_DIR / "combined_knn.png"
    plt.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
