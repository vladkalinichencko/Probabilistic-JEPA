#!/usr/bin/env python3
"""
Compare latent distributions produced by three predictors:
  - Deterministic i-JEPA baseline (single-mode).
  - Multivariate MDN head.
  - Autoregressive (RNADE-style) head.

For each predictor we sample target embeddings for a validation batch,
draw samples from the predictive distribution (where applicable), and
project them with PCA for qualitative 2D/3D visualization.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _import_from(path: Path, module_name: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_masks(grid: int, sampler):
    ctx_mask, tgt_mask = sampler(grid)
    return ctx_mask.unsqueeze(0), tgt_mask.unsqueeze(0)


def _project(samples: np.ndarray, targets: np.ndarray, dims: int = 3):
    joined = np.concatenate([samples, targets], axis=0)
    pca = PCA(n_components=min(dims, joined.shape[1]))
    proj = pca.fit_transform(joined)
    return proj[: samples.shape[0]], proj[samples.shape[0] :], pca


def _hexbin(ax, pts, teachers, title):
    hb = ax.hexbin(pts[:, 0], pts[:, 1], gridsize=45, cmap="magma", mincnt=1)
    ax.scatter(teachers[:, 0], teachers[:, 1], c="cyan", s=12, label="Teacher", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    return hb


def _scatter_3d(ax, pts, teachers, title):
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=6,
        c=pts[:, 2],
        cmap="viridis",
        alpha=0.35,
    )
    ax.scatter(
        teachers[:, 0],
        teachers[:, 1],
        teachers[:, 2],
        c="cyan",
        s=25,
        label="Teacher",
        depthshade=False,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=25, azim=35)


def sample_mdn(draws_per_token: int = 256):
    module = _import_from(ROOT / "MDN" / "mdn.py", "mdn_module")
    device = _device()
    ckpt = torch.load(ROOT / "MDN" / "mdn_last.pt", map_location=device)
    student = module.Student(
        module.ViTBackbone(
            module.IMAGE_SIZE,
            module.PATCH_SIZE,
            module.HIDDEN_DIM,
            module.VIT_LAYERS,
            module.VIT_HEADS,
        )
    ).to(device)
    teacher = module.Teacher(
        module.ViTBackbone(
            module.IMAGE_SIZE,
            module.PATCH_SIZE,
            module.HIDDEN_DIM,
            module.VIT_LAYERS,
            module.VIT_HEADS,
        )
    ).to(device)
    predictor = module.MDNPredictor(module.HIDDEN_DIM, num_layers=2, num_heads=module.VIT_HEADS).to(device)
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    predictor.load_state_dict(ckpt["predictor"])
    student.eval()
    teacher.eval()
    predictor.eval()

    dataset = module.get_dataset("valid")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)
    ctx_mask, tgt_mask = module.sample_context_and_targets(student.backbone.grid)
    ctx_mask = ctx_mask.unsqueeze(0).expand(x.size(0), -1).to(device)
    tgt_mask = tgt_mask.unsqueeze(0).expand(x.size(0), -1).to(device)

    with torch.no_grad():
        teacher_tokens = teacher(x)
    idx = [tgt_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(x.size(0))]
    St = int(max([int(i.numel()) for i in idx]))
    D = teacher_tokens.size(2)
    teacher_tgt = torch.zeros((x.size(0), St, D), device=device, dtype=teacher_tokens.dtype)
    tgt_pad = torch.ones((x.size(0), St), dtype=torch.bool, device=device)
    pos = student.backbone.pos_embed
    tgt_q = torch.zeros_like(teacher_tgt)
    for i, ids in enumerate(idx):
        L = int(ids.numel())
        if L == 0:
            continue
        teacher_tgt[i, :L] = teacher_tokens[i, ids]
        tgt_pad[i, :L] = False
        tgt_q[i, :L] = predictor.q_token[None, :].expand(L, -1) + pos[0, ids]

    ctx_tokens, ctx_pad = student.encode_context(x, ctx_mask)
    with torch.no_grad():
        pi, mu, sigma = predictor(ctx_tokens, ctx_pad, tgt_q, tgt_pad)

    pi_flat = pi[~tgt_pad]
    mu_flat = mu[~tgt_pad]  # [N, K, D]
    sigma_flat = sigma[~tgt_pad]
    teacher_flat = teacher_tgt[~tgt_pad].cpu().numpy()
    num_tokens = pi_flat.size(0)
    samples = []
    for n in range(num_tokens):
        comp = torch.distributions.Categorical(pi_flat[n]).sample((draws_per_token,))
        chosen_mu = mu_flat[n][comp, :]
        chosen_sigma = sigma_flat[n][comp, :]
        eps = torch.randn_like(chosen_mu)
        draw = chosen_mu + chosen_sigma * eps
        samples.append(draw)
    samples = torch.cat(samples, dim=0).cpu().numpy()
    return samples, teacher_flat


def sample_autoregressive(draws_per_token: int = 64):
    device = _device()
    arn = _import_from(ROOT / "autoregression" / "rnade_model.py", "rnade_module")
    ckpt_student = torch.load(ROOT / "autoregression" / "student_model.pth", map_location=device)
    ckpt_teacher = torch.load(ROOT / "autoregression" / "teacher_model.pth", map_location=device)
    ckpt_predictor = torch.load(ROOT / "autoregression" / "predictor_model.pth", map_location=device)

    student = arn.Student(
        arn.ViTBackbone(
            arn.IMAGE_SIZE,
            arn.PATCH_SIZE,
            arn.HIDDEN_DIM,
            arn.VIT_LAYERS,
            arn.VIT_HEADS,
        )
    ).to(device)
    teacher = arn.Teacher(
        arn.ViTBackbone(
            arn.IMAGE_SIZE,
            arn.PATCH_SIZE,
            arn.HIDDEN_DIM,
            arn.VIT_LAYERS,
            arn.VIT_HEADS,
        )
    ).to(device)
    predictor = arn.MaskedAutoregressiveMLPConditional(
        input_dim=arn.HIDDEN_DIM,
        hidden_dims=[1024, 768, 512],
        num_components=arn.NUM_COMPONENTS,
        context_dim=arn.HIDDEN_DIM,
    ).to(device)
    student.load_state_dict(ckpt_student)
    teacher.load_state_dict(ckpt_teacher)
    predictor.load_state_dict(ckpt_predictor)
    student.eval()
    teacher.eval()
    predictor.eval()

    dataset = arn.get_dataset("valid")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)
    grid = student.backbone.grid
    ctx_mask, tgt_mask = arn.sample_context_and_targets(grid)
    ctx_mask = ctx_mask.unsqueeze(0).to(device)
    tgt_mask = tgt_mask.unsqueeze(0).to(device)

    ctx_out, ctx_pad = student.encode_context(x, ctx_mask)
    mask_f = (~ctx_pad).float().unsqueeze(-1)
    ctx_mean = (ctx_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    ctx_feat = torch.layer_norm(ctx_mean, ctx_mean.shape[-1:])

    with torch.no_grad():
        teacher_tokens = teacher(x)
    idx = tgt_mask[0].nonzero(as_tuple=False).squeeze(1)
    St = int(idx.numel())
    D = teacher_tokens.size(2)
    teacher_tgt = teacher_tokens[:, idx, :]
    teacher_flat = teacher_tgt.reshape(-1, D)
    mean = teacher_flat.mean(dim=0, keepdim=True)
    std = teacher_flat.std(dim=0, keepdim=True) + 1e-6
    teacher_norm = (teacher_flat - mean) / std
    ctx_rep = ctx_feat.unsqueeze(1).expand(1, St, -1).reshape(St, -1)
    with torch.no_grad():
        alpha, mu, sigma = predictor(teacher_norm, ctx_rep)

    samples = []
    alpha_np = alpha.cpu().numpy()
    mu_np = mu.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    K = alpha_np.shape[-1]
    samples = []
    for token_idx in range(alpha_np.shape[0]):
        for _ in range(draws_per_token):
            dims = []
            for d in range(alpha_np.shape[1]):
                comp = np.random.choice(K, p=alpha_np[token_idx, d])
                val = np.random.normal(mu_np[token_idx, d, comp], sigma_np[token_idx, d, comp])
                dims.append(val)
            samples.append(dims)
    samples = np.array(samples, dtype=np.float32)
    teacher_np = teacher_norm.cpu().numpy()
    samples = samples * std.cpu().numpy() + mean.cpu().numpy()
    teacher_np = teacher_np * std.cpu().numpy() + mean.cpu().numpy()
    return samples, teacher_np


def sample_baseline():
    module = _import_from(ROOT / "I-JEPA" / "i-jepa.py", "ijepa_eval_module")
    candidates = [
        ROOT / "I-JEPA" / "checkpoints" / "ijeja_best_knn.pt",
        ROOT.parent / "BionicEye" / "representation" / "current" / "checkpoints" / "ijeja_best_knn.pt",
        ROOT.parent / "BionicEye" / "representation" / "checkpoints" / "ijeja_best_knn.pt",
    ]
    ckpt_path = None
    for cand in candidates:
        if cand.exists():
            ckpt_path = cand
            break
    if ckpt_path is None:
        raise FileNotFoundError("Baseline checkpoint not found")
    device = _device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hparams") or {}
    image_size = int(getattr(module, "IMAGE_SIZE", 64))
    patch_size = int(hparams.get("PATCH_SIZE", getattr(module, "PATCH_SIZE", 8)))
    hidden_dim = int(hparams.get("HIDDEN_DIM", getattr(module, "HIDDEN_DIM", 384)))
    vit_layers = int(hparams.get("VIT_LAYERS", getattr(module, "VIT_LAYERS", 10)))
    vit_heads = int(hparams.get("VIT_HEADS", getattr(module, "VIT_HEADS", 8)))
    student = module.Student(
        module.ViTBackbone(
            image_size,
            patch_size,
            hidden_dim,
            vit_layers,
            vit_heads,
        )
    ).to(device)
    teacher = module.Teacher(
        module.ViTBackbone(
            image_size,
            patch_size,
            hidden_dim,
            vit_layers,
            vit_heads,
        )
    ).to(device)
    predictor = module.Predictor(hidden_dim, num_layers=2, num_heads=vit_heads).to(device)
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    predictor.load_state_dict(ckpt["predictor"])
    student.eval()
    teacher.eval()
    predictor.eval()

    dataset = module.get_dataset("valid")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)
    ctx_mask, tgt_mask = module.sample_context_and_targets(student.backbone.grid)
    ctx_mask = ctx_mask.unsqueeze(0).expand(x.size(0), -1).to(device)
    tgt_mask = tgt_mask.unsqueeze(0).expand(x.size(0), -1).to(device)

    with torch.no_grad():
        teacher_tokens = teacher(x)
    idx = [tgt_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(x.size(0))]
    St = int(max([int(i.numel()) for i in idx]))
    D = teacher_tokens.size(2)
    teacher_tgt = torch.zeros((x.size(0), St, D), device=device, dtype=teacher_tokens.dtype)
    tgt_pad = torch.ones((x.size(0), St), dtype=torch.bool, device=device)
    pos = student.backbone.pos_embed
    tgt_q = torch.zeros_like(teacher_tgt)
    for i, ids in enumerate(idx):
        L = int(ids.numel())
        if L == 0:
            continue
        teacher_tgt[i, :L] = teacher_tokens[i, ids]
        tgt_pad[i, :L] = False
        tgt_q[i, :L] = predictor.q_token[None, :].expand(L, -1) + pos[0, ids]
    ctx_tokens, ctx_pad = student.encode_context(x, ctx_mask)
    with torch.no_grad():
        pred_tgt = predictor(ctx_tokens, ctx_pad, tgt_q, tgt_pad)
    samples = pred_tgt[~tgt_pad].cpu().numpy()
    teacher_np = teacher_tgt[~tgt_pad].cpu().numpy()
    return samples, teacher_np


def main():
    mdn_samples, mdn_teacher = sample_mdn()
    arn_samples, arn_teacher = sample_autoregressive()
    base_samples, base_teacher = sample_baseline()

    models = [
        ("MDN", mdn_samples, mdn_teacher, FIG_DIR / "mdn_distribution.png"),
        ("Autoregressive", arn_samples, arn_teacher, FIG_DIR / "autoregressive_distribution.png"),
        ("Deterministic", base_samples, base_teacher, FIG_DIR / "baseline_distribution.png"),
    ]

    # Combined comparison figure
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(len(models), 2)
    for idx, (name, samples, teacher, _) in enumerate(models):
        sample_2d, teacher_2d, _ = _project(samples, teacher, dims=2)
        sample_3d, teacher_3d, _ = _project(samples, teacher, dims=3)
        ax2d = fig.add_subplot(gs[idx, 0])
        hb = _hexbin(ax2d, sample_2d, teacher_2d, f"{name} PCA-2D density")
        fig.colorbar(hb, ax=ax2d, fraction=0.046, pad=0.04)
        ax3d = fig.add_subplot(gs[idx, 1], projection="3d")
        _scatter_3d(ax3d, sample_3d, teacher_3d, f"{name} PCA-3D view")
    plt.tight_layout()
    out_path = FIG_DIR / "distribution_comparison.png"
    plt.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")

    # Per-model 2D histograms (for case-study sections)
    for name, samples, teacher, path in models:
        sample_2d, teacher_2d, _ = _project(samples, teacher, dims=2)
        plt.figure(figsize=(6, 5))
        hb = plt.hexbin(
            sample_2d[:, 0],
            sample_2d[:, 1],
            gridsize=50,
            cmap="magma",
            mincnt=1,
        )
        plt.scatter(
            teacher_2d[:, 0],
            teacher_2d[:, 1],
            c="cyan",
            s=10,
            alpha=0.6,
            label="Teacher tokens",
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{name} latent distribution (PCA-2D hexbin)")
        plt.grid(alpha=0.25)
        plt.legend(loc="upper right")
        plt.colorbar(hb, label="sample density")
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
