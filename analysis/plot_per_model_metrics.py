#!/usr/bin/env python3
"""
Produce per-model loss and kNN figures:
- loss_baseline.png, loss_mdn_placeholder.png, loss_autoregressive_placeholder.png,
  loss_flow.png, loss_diffusion.png
- knn_baseline.png, knn_mdn.png, knn_autoregressive.png, knn_flow.png, knn_diffusion.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "analysis" / "cache"


def load_baseline_loss():
    path = CACHE_DIR / "ijepa_val_loss.json"
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    return data.get("steps", []), data.get("total_loss", [])


def load_diffusion_loss():
    scalars_path = ROOT / "diffusion" / "scalars (2).json"
    if not scalars_path.exists():
        return [], [], [], []
    data = json.loads(scalars_path.read_text())
    t_steps = [r["step"] for r in data.get("train/loss", [])]
    t_vals = [r["value"] for r in data.get("train/loss", [])]
    v_steps = [r["step"] for r in data.get("val/loss", [])]
    v_vals = [r["value"] for r in data.get("val/loss", [])]
    return t_steps, t_vals, v_steps, v_vals


def load_flow_loss():
    # take best run with both train and val loss
    base = ROOT / "flow_matching" / "runs"
    best = None
    best_last_step = -1
    for event_path in sorted(base.glob("*/events.out.tfevents.*")):
        acc = EventAccumulator(str(event_path))
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if "train/loss" not in tags and "train/nll" not in tags:
            continue
        if "val/loss" not in tags and "val/nll" not in tags:
            continue
        tag_train = "train/nll" if "train/nll" in tags else "train/loss"
        tag_val = "val/nll" if "val/nll" in tags else "val/loss"

        train_events = acc.Scalars(tag_train)
        val_events = acc.Scalars(tag_val)
        if not train_events or not val_events:
            continue
        first_step = train_events[0].step
        if first_step > 5_000_000:
            continue
        last_step = train_events[-1].step
        if last_step > best_last_step:
            best_last_step = last_step
            best = (train_events, val_events)

    if best is None:
        return [], [], [], []
    train_events, val_events = best
    t_steps = [e.step for e in train_events]
    t_vals = [e.value for e in train_events]
    v_steps = [e.step for e in val_events]
    v_vals = [e.value for e in val_events]
    return t_steps, t_vals, v_steps, v_vals


def load_knn_cache():
    path = CACHE_DIR / "knn_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_diffusion_knn():
    scalars_path = ROOT / "diffusion" / "scalars (2).json"
    if not scalars_path.exists():
        return [], []
    data = json.loads(scalars_path.read_text())
    steps = [r["step"] for r in data.get("val/knn_top1", [])]
    vals = [r["value"] * 100.0 for r in data.get("val/knn_top1", [])]
    return steps, vals


def load_flow_knn():
    base = ROOT / "flow_matching" / "runs"
    data = {}
    for event_path in sorted(base.glob("*/events.out.tfevents.*")):
        acc = EventAccumulator(str(event_path))
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if "val/knn_top1" not in tags:
            continue
        events = acc.Scalars("val/knn_top1")
        if not events:
            continue
        if events[0].step > 5_000_000:
            continue
        for e in events:
            data[e.step] = e.value * 100.0
    steps = sorted(data)
    vals = [data[s] for s in steps]
    return steps, vals


def plot_baseline_loss():
    steps, vals = load_baseline_loss()
    plt.figure(figsize=(4.8, 3.2))
    if steps:
        plt.plot(np.array(steps) / 1e6, vals, label="val loss")
    plt.xlabel("Samples seen (millions)")
    plt.ylabel("Loss")
    plt.title("Baseline I-JEPA validation loss")
    plt.grid(alpha=0.3)
    if steps:
        plt.legend()
    out = FIG_DIR / "loss_baseline.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_flow_loss():
    t_steps, t_vals, v_steps, v_vals = load_flow_loss()
    plt.figure(figsize=(4.8, 3.2))
    if t_steps:
        plt.plot(np.array(t_steps) / 1e6, t_vals, label="train NLL")
    if v_steps:
        plt.plot(np.array(v_steps) / 1e6, v_vals, label="val NLL")
    plt.xlabel("Steps (millions)")
    plt.ylabel("NLL")
    plt.title("Flow-matching loss")
    plt.grid(alpha=0.3)
    if t_steps or v_steps:
        plt.legend()
    out = FIG_DIR / "loss_flow.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_diffusion_loss():
    t_steps, t_vals, v_steps, v_vals = load_diffusion_loss()
    plt.figure(figsize=(4.8, 3.2))
    if t_steps:
        plt.plot(np.array(t_steps) / 1e6, t_vals, label="train loss")
    if v_steps:
        plt.plot(np.array(v_steps) / 1e6, v_vals, label="val loss")
    plt.xlabel("Steps (millions)")
    plt.ylabel("Loss")
    plt.title("Diffusion loss")
    plt.grid(alpha=0.3)
    if t_steps or v_steps:
        plt.legend()
    out = FIG_DIR / "loss_diffusion.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_placeholder_loss(name: str, title: str, filename: str):
    plt.figure(figsize=(4.8, 3.2))
    plt.axis("off")
    plt.text(
        0.5,
        0.5,
        f"{title}\n\n(placeholder: loss curve not logged\nfor this head in current repo.)",
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
    )
    out = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_knn_baseline_mdn_autoreg():
    cache = load_knn_cache()
    baseline = float(cache.get("baseline", 0.0))
    mdn = float(cache.get("mdn", 0.0))
    arn = float(cache.get("autoregressive", 0.0))

    def _flat(name: str, val: float, filename: str):
        plt.figure(figsize=(4.8, 3.2))
        if val > 0:
            plt.axhline(val, color="tab:blue")
            plt.text(0.05, val + 0.3, f"{val:.2f}%", color="tab:blue")
        plt.xlabel("Relative training progress")
        plt.ylabel("kNN@1 (%)")
        plt.title(f"{name} validation kNN@1")
        plt.grid(alpha=0.3)
        out = FIG_DIR / filename
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()

    _flat("Baseline I-JEPA", baseline, "knn_baseline.png")
    _flat("MDN", mdn, "knn_mdn.png")
    _flat("Autoregressive", arn, "knn_autoregressive.png")


def plot_knn_flow_diffusion():
    f_steps, f_vals = load_flow_knn()
    d_steps, d_vals = load_diffusion_knn()

    plt.figure(figsize=(4.8, 3.2))
    if f_steps:
        plt.plot(np.array(f_steps) / 1e6, f_vals)
    plt.xlabel("Steps (millions)")
    plt.ylabel("kNN@1 (%)")
    plt.title("Flow-matching validation kNN@1")
    plt.grid(alpha=0.3)
    out = FIG_DIR / "knn_flow.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()

    plt.figure(figsize=(4.8, 3.2))
    if d_steps:
        plt.plot(np.array(d_steps) / 1e6, d_vals)
    plt.xlabel("Steps (millions)")
    plt.ylabel("kNN@1 (%)")
    plt.title("Diffusion validation kNN@1")
    plt.grid(alpha=0.3)
    out = FIG_DIR / "knn_diffusion.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def main():
    # Losses
    plot_baseline_loss()
    plot_flow_loss()
    plot_diffusion_loss()
    plot_placeholder_loss("MDN", "MDN loss", "loss_mdn_placeholder.png")
    plot_placeholder_loss("Autoregressive", "Autoregressive loss", "loss_autoregressive_placeholder.png")

    # kNN
    plot_knn_baseline_mdn_autoreg()
    plot_knn_flow_diffusion()


if __name__ == "__main__":
    main()

