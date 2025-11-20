#!/usr/bin/env python3
"""
Evaluate the deterministic I-JEPA baseline checkpoint and export per-batch
validation losses so the curves can be plotted alongside probabilistic heads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "analysis" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(path: Path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("ijepa_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _find_checkpoint(explicit: str | None = None) -> Path:
    if explicit:
        ckpt = Path(explicit).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt
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
    raise FileNotFoundError(
        "Could not locate i-JEPA checkpoint. Pass --ckpt manually."
    )


def evaluate(ckpt_path: Path, max_batches: int | None = None) -> dict:
    mod = _load_module(ROOT / "I-JEPA" / "i-jepa.py")
    device = mod.DEVICE
    dataset = mod.get_dataset("valid")
    loader = DataLoader(
        dataset,
        batch_size=mod.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=mod.PIN_MEMORY,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hparams") or {}
    image_size = int(getattr(mod, "IMAGE_SIZE", 64))
    patch_size = int(hparams.get("PATCH_SIZE", getattr(mod, "PATCH_SIZE", 8)))
    hidden_dim = int(hparams.get("HIDDEN_DIM", getattr(mod, "HIDDEN_DIM", 384)))
    vit_layers = int(hparams.get("VIT_LAYERS", getattr(mod, "VIT_LAYERS", 10)))
    vit_heads = int(hparams.get("VIT_HEADS", getattr(mod, "VIT_HEADS", 8)))

    student = mod.Student(
        mod.ViTBackbone(
            image_size,
            patch_size,
            hidden_dim,
            vit_layers,
            vit_heads,
        )
    ).to(device)
    teacher = mod.Teacher(
        mod.ViTBackbone(
            image_size,
            patch_size,
            hidden_dim,
            vit_layers,
            vit_heads,
        )
    ).to(device)
    predictor = mod.Predictor(hidden_dim, num_layers=2, num_heads=vit_heads).to(
        device
    )
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    predictor.load_state_dict(ckpt["predictor"])
    student.eval()
    teacher.eval()
    predictor.eval()

    samples_base = int(ckpt.get("samples_seen", 0) or 0)
    steps = []
    total_losses = []
    align_losses = []
    var_losses = []
    cov_losses = []
    processed = 0

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            B = x.size(0)
            grid = student.backbone.grid
            ctx_list = []
            tgt_list = []
            for _ in range(B):
                ctx_mask, tgt_mask = mod.sample_context_and_targets(grid)
                ctx_list.append(ctx_mask)
                tgt_list.append(tgt_mask)
            context_mask = torch.stack(ctx_list, dim=0).to(device)
            target_mask = torch.stack(tgt_list, dim=0).to(device)

            t_tokens = teacher(x)
            tgt_idxs = [target_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(B)]
            St = int(max([int(idx.numel()) for idx in tgt_idxs]) if tgt_idxs else 0)
            D = t_tokens.size(2)
            teacher_tgt = torch.zeros((B, St, D), device=device, dtype=t_tokens.dtype)
            tgt_pad = torch.ones((B, St), dtype=torch.bool, device=device)
            tgt_q = torch.zeros((B, St, D), device=device, dtype=t_tokens.dtype)
            pos = student.backbone.pos_embed
            for i, idx in enumerate(tgt_idxs):
                L = int(idx.numel())
                if L == 0:
                    continue
                teacher_tgt[i, :L] = t_tokens[i, idx]
                tgt_pad[i, :L] = False
                tgt_q[i, :L] = predictor.q_token[None, :].expand(L, -1) + pos[0, idx]

            ctx_tokens, ctx_pad = student.encode_context(x, context_mask)
            pred_tgt = predictor(ctx_tokens, ctx_pad, tgt_q, tgt_pad)
            align, var, cov = mod.jepa_loss_components(
                pred_tgt, teacher_tgt, tgt_pad, std_floor=mod.LOSS_STD_FLOOR
            )
            loss = (
                mod.LOSS_ALIGN_WEIGHT * align
                + mod.LOSS_VAR_WEIGHT * var
                + mod.LOSS_COV_WEIGHT * cov
            )

            processed += B
            steps.append(samples_base + processed)
            total_losses.append(float(loss.item()))
            align_losses.append(float(align.item()))
            var_losses.append(float(var.item()))
            cov_losses.append(float(cov.item()))

    return {
        "steps": steps,
        "total_loss": total_losses,
        "align": align_losses,
        "var": var_losses,
        "cov": cov_losses,
        "samples_seen_start": samples_base,
        "best_knn": float(ckpt.get("best_knn", 0.0) or 0.0),
        "checkpoint": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to i-JEPA checkpoint (defaults to repo/BionicEye versions).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of validation batches to process.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(CACHE_DIR / "ijepa_val_loss.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()
    ckpt_path = _find_checkpoint(args.ckpt)
    data = evaluate(ckpt_path, max_batches=args.max_batches)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved baseline loss trace → {out_path}")


if __name__ == "__main__":
    main()
