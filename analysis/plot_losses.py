#!/usr/bin/env python3
"""
Aggregate diffusion and flow-matching training curves into a single figure.

Outputs report/figures/combined_losses.png (train/val loss or NLL).
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "analysis" / "cache"


def load_diffusion_losses():
    scalars_path = ROOT / "diffusion" / "scalars (2).json"
    train_steps, train_loss = [], []
    val_steps, val_loss = [], []
    if not scalars_path.exists():
        return train_steps, train_loss, val_steps, val_loss
    data = json.loads(scalars_path.read_text())
    for rec in data.get("train/loss", []):
        train_steps.append(rec["step"])
        train_loss.append(rec["value"])
    for rec in data.get("val/loss", []):
        val_steps.append(rec["step"])
        val_loss.append(rec["value"])
    return train_steps, train_loss, val_steps, val_loss


def _aggregate_flow_scalars(tag: str, start_threshold: int = 5_000_000, min_points: int = 15):
    data = {}
    base = ROOT / "flow_matching" / "runs"
    event_files = sorted(base.glob("*/events.out.tfevents.*"))
    if not event_files:
        return [], []
    for event_path in event_files:
        acc = EventAccumulator(str(event_path))
        acc.Reload()
        scalars = acc.Tags().get("scalars", [])
        if tag not in scalars:
            continue
        events = acc.Scalars(tag)
        if not events:
            continue
        steps = [ev.step for ev in events]
        if steps[0] > start_threshold or len(steps) < min_points:
            continue
        for ev in events:
            data[ev.step] = ev.value
    steps = sorted(data)
    values = [data[s] for s in steps]
    return steps, values


def load_flow_losses():
    train_steps, train_vals = _aggregate_flow_scalars("train/nll")
    val_steps, val_vals = _aggregate_flow_scalars("val/nll")
    return train_steps, train_vals, val_steps, val_vals


def load_baseline_losses():
    cache_path = CACHE_DIR / "ijepa_val_loss.json"
    if not cache_path.exists():
        return [], [], [], [], []
    data = json.loads(cache_path.read_text())
    steps = data.get("steps", [])
    total = data.get("total_loss", [])
    align = data.get("align", [])
    var = data.get("var", [])
    cov = data.get("cov", [])
    return steps, total, align, var, cov


def main():
    diff_train_steps, diff_train_loss, diff_val_steps, diff_val_loss = load_diffusion_losses()
    flow_train_steps, flow_train_nll, flow_val_steps, flow_val_nll = load_flow_losses()
    base_steps, base_total, base_align, base_var, base_cov = load_baseline_losses()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    if base_steps:
        axes[0].plot(np.array(base_steps) / 1e6, base_total, label="val total", color="tab:blue")
        axes[0].plot(np.array(base_steps) / 1e6, base_align, label="align", color="tab:orange", alpha=0.7)
        axes[0].plot(np.array(base_steps) / 1e6, base_var, label="variance", color="tab:green", alpha=0.7)
        axes[0].plot(np.array(base_steps) / 1e6, base_cov, label="covariance", color="tab:red", alpha=0.7)
    axes[0].set_title("Deterministic I-JEPA (val)")
    axes[0].set_xlabel("Samples processed (millions)")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    if base_steps:
        axes[0].legend()

    if diff_train_steps:
        axes[1].plot(np.array(diff_train_steps) / 1e6, diff_train_loss, label="train loss")
    if diff_val_steps:
        axes[1].plot(np.array(diff_val_steps) / 1e6, diff_val_loss, label="val loss")
    axes[1].set_title("Diffusion losses")
    axes[1].set_xlabel("Steps (millions)")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if flow_train_steps:
        axes[2].plot(np.array(flow_train_steps) / 1e6, flow_train_nll, label="train loss")
    if flow_val_steps:
        axes[2].plot(np.array(flow_val_steps) / 1e6, flow_val_nll, label="val loss")
    axes[2].set_title("Flow-matching losses")
    axes[2].set_xlabel("Steps (millions)")
    axes[2].set_ylabel("Loss (NLL)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    out_path = FIG_DIR / "combined_losses.png"
    plt.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
