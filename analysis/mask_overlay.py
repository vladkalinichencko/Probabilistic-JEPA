#!/usr/bin/env python3
"""
Visualize the block masking strategy on a Tiny ImageNet sample.

Outputs report/figures/mask_overlay.png.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from MDN.mdn import (
    ViTBackbone,
    sample_context_and_targets,
    IMAGE_SIZE,
    PATCH_SIZE,
    HIDDEN_DIM,
    VIT_LAYERS,
    VIT_HEADS,
)


FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    backbone = ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)
    grid = backbone.grid

    dataset = load_dataset("zh-plus/tiny-imagenet", split="valid").filter(lambda e: int(e["label"]) < 100)
    img = dataset[1]["image"]
    tfm = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])
    img_arr = np.array(tfm(img))

    ctx_mask, tgt_mask = sample_context_and_targets(grid)
    ctx_mask = ctx_mask.reshape(grid, grid)
    tgt_mask = tgt_mask.reshape(grid, grid)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_arr)
    for y in range(grid + 1):
        ax.axhline(y * IMAGE_SIZE / grid, color="white", alpha=0.2, linewidth=0.5)
    for x in range(grid + 1):
        ax.axvline(x * IMAGE_SIZE / grid, color="white", alpha=0.2, linewidth=0.5)
    for y in range(grid):
        for x in range(grid):
            y0, y1 = y * IMAGE_SIZE / grid, (y + 1) * IMAGE_SIZE / grid
            x0, x1 = x * IMAGE_SIZE / grid, (x + 1) * IMAGE_SIZE / grid
            if ctx_mask[y, x]:
                rect = plt.Rectangle((x0, y0), IMAGE_SIZE / grid, IMAGE_SIZE / grid, linewidth=1, edgecolor="lime", facecolor="lime", alpha=0.25)
                ax.add_patch(rect)
            if tgt_mask[y, x]:
                rect = plt.Rectangle((x0, y0), IMAGE_SIZE / grid, IMAGE_SIZE / grid, linewidth=1, edgecolor="red", facecolor="red", alpha=0.25)
                ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Context (green) vs target (red) patches")
    plt.tight_layout()
    out_path = FIG_DIR / "mask_overlay.png"
    plt.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
