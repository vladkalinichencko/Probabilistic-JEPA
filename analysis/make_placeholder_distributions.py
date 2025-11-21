#!/usr/bin/env python3
"""
Create simple placeholder distribution figures for flow and diffusion heads.

These are intentionally marked as placeholders in the report; they should be
replaced with real latent-distribution visualizations once the final
checkpoints for these models are available in the repository.
"""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _make_placeholder(name: str, text: str):
    plt.figure(figsize=(4.8, 3.6))
    plt.axis("off")
    plt.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
    )
    out = FIG_DIR / f"{name}_distribution_placeholder.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Saved {out}")


def main():
    _make_placeholder(
        "flow",
        "Flow-matching latent distribution\n\n"
        "(placeholder figure: insert PCA-based\n"
        "histogram of flow samples once checkpoints\n"
        "are exported to this repository).",
    )
    _make_placeholder(
        "diffusion",
        "Diffusion latent distribution\n\n"
        "(placeholder figure: insert PCA-based\n"
        "histogram of diffusion samples once\n"
        "checkpoints are available).",
    )


if __name__ == "__main__":
    main()

