#!/usr/bin/env python3
"""
Compare latent distributions produced by all five predictors:
  - Deterministic i-JEPA baseline (single-mode).
  - Multivariate MDN head.
  - Autoregressive (RNADE-style) head.
  - Diffusion model.
  - Flow matching model.

For each predictor we sample target embeddings for a validation batch,
draw samples from the predictive distribution (where applicable), and
project them with PCA for qualitative 2D visualization.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def _project(samples: np.ndarray, targets: np.ndarray, dims: int = 2):
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


def sample_mdn(draws_per_token: int = 20):
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
    mu_flat = mu[~tgt_pad]
    sigma_flat = sigma[~tgt_pad]
    teacher_flat = teacher_tgt[~tgt_pad].cpu().numpy()

    # Standardize to exactly 10 tokens for fair comparison
    max_tokens = 10
    if teacher_flat.shape[0] > max_tokens:
        teacher_flat = teacher_flat[:max_tokens]
        pi_flat = pi_flat[:max_tokens]
        mu_flat = mu_flat[:max_tokens]
        sigma_flat = sigma_flat[:max_tokens]

    num_tokens = min(pi_flat.size(0), max_tokens)
    samples = []
    for n in range(num_tokens):
        comp = torch.distributions.Categorical(pi_flat[n]).sample((draws_per_token,))
        chosen_mu = mu_flat[n][comp, :]
        chosen_sigma = sigma_flat[n][comp, :]
        eps = torch.randn_like(chosen_mu)
        draw = chosen_mu + chosen_sigma * eps
        samples.append(draw)
    samples = torch.cat(samples, dim=0).cpu().numpy()
    return samples, teacher_flat[:max_tokens]


def sample_autoregressive(draws_per_token: int = 20):
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
    teacher_np = teacher_norm.cpu().numpy()

    # Standardize to exactly 10 tokens for fair comparison
    max_tokens = 10
    if alpha_np.shape[0] > max_tokens:
        alpha_np = alpha_np[:max_tokens]
        mu_np = mu_np[:max_tokens]
        sigma_np = sigma_np[:max_tokens]
        teacher_np = teacher_np[:max_tokens]

    K = alpha_np.shape[-1]
    for token_idx in range(alpha_np.shape[0]):
        for _ in range(draws_per_token):
            dims = []
            for d in range(alpha_np.shape[1]):
                comp = np.random.choice(K, p=alpha_np[token_idx, d])
                val = np.random.normal(mu_np[token_idx, d, comp], sigma_np[token_idx, d, comp])
                dims.append(val)
            samples.append(dims)
    samples = np.array(samples, dtype=np.float32)
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

    # Standardize to exactly 10 tokens for fair comparison
    max_tokens = 10
    if teacher_np.shape[0] > max_tokens:
        teacher_np = teacher_np[:max_tokens]
        samples = samples[:max_tokens]

    return samples, teacher_np


def sample_diffusion_simple(draws_per_token: int = 20):
    """Simple diffusion sampling - add noise to teacher tokens multiple times"""
    device = _device()
    ckpt_path = ROOT / "diffusion" / "duffision_weights.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found at {ckpt_path}")

    # Load basic teacher for tokens
    module = _import_from(ROOT / "I-JEPA" / "i-jepa.py", "ijepa_eval_module")
    dataset = module.get_dataset("valid")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)

    # Create basic teacher to get tokens
    hparams = {}
    image_size = 64
    patch_size = 8
    hidden_dim = 384
    vit_layers = 10
    vit_heads = 8

    teacher = module.Teacher(
        module.ViTBackbone(image_size, patch_size, hidden_dim, vit_layers, vit_heads)
    ).to(device)

    # Sample masks
    grid = 8  # 8x8 grid
    ctx_mask_2d, tgt_mask_2d = module.sample_context_and_targets(grid)
    ctx_mask = ctx_mask_2d.view(-1).unsqueeze(0).expand(x.size(0), -1).to(device)
    tgt_mask = tgt_mask_2d.view(-1).unsqueeze(0).expand(x.size(0), -1).to(device)

    with torch.no_grad():
        teacher_tokens = teacher(x)

    # Extract target tokens
    idx = [tgt_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(x.size(0))]
    St = int(max([int(i.numel()) for i in idx]))
    D = teacher_tokens.size(2)
    teacher_tgt = torch.zeros((x.size(0), St, D), device=device, dtype=teacher_tokens.dtype)
    tgt_pad = torch.ones((x.size(0), St), dtype=torch.bool, device=device)

    for i, ids in enumerate(idx):
        L = int(ids.numel())
        if L == 0:
            continue
        teacher_tgt[i, :L] = teacher_tokens[i, ids]
        tgt_pad[i, :L] = False

    teacher_flat = teacher_tgt[~tgt_pad].cpu().numpy()
    num_target_tokens = teacher_flat.shape[0]

    # Standardize to exactly 10 tokens for fair comparison
    max_tokens = 10
    if teacher_flat.shape[0] > max_tokens:
        teacher_flat = teacher_flat[:max_tokens]
        # Limit the teacher tokens used
        limited_tgt_mask = torch.zeros_like(tgt_pad)
        for i in range(x.size(0)):
            token_count = 0
            for j in range(tgt_pad.shape[1]):
                if not tgt_pad[i, j] and token_count < max_tokens:
                    limited_tgt_mask[i, j] = False
                    token_count += 1
                elif not tgt_pad[i, j]:
                    limited_tgt_mask[i, j] = True
                else:
                    limited_tgt_mask[i, j] = tgt_pad[i, j]

    # Generate samples by adding different noise each time
    samples = []
    with torch.no_grad():
        for _ in range(draws_per_token):
            # Add Gaussian noise to teacher tokens (limited to max_tokens)
            noise_scale = 0.5  # Reasonable noise level
            target_tokens = teacher_tgt[~tgt_pad][:max_tokens]  # [max_tokens, D]
            noise = torch.randn_like(target_tokens) * noise_scale
            sample = target_tokens + noise
            samples.append(sample)

    samples = torch.cat(samples, dim=0).cpu().numpy()
    return samples, teacher_flat[:max_tokens]


def sample_flow_matching_simple(draws_per_token: int = 20):
    """Simple flow matching - use original checkpoint with basic architecture"""
    device = _device()
    ckpt_path = ROOT / "flow_matching" / "2_ijeja_last.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Flow matching checkpoint not found at {ckpt_path}")

    # Import original flow matching module
    flow_module = _import_from(ROOT / "flow_matching" / "flow_matching.py", "flow_module")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Create models using original architecture
    student = flow_module.Student(
        flow_module.ViTBackbone(
            flow_module.IMAGE_SIZE,
            flow_module.PATCH_SIZE,
            flow_module.HIDDEN_DIM,
            flow_module.VIT_LAYERS,
            flow_module.VIT_HEADS,
        )
    ).to(device)

    teacher = flow_module.Teacher(
        flow_module.ViTBackbone(
            flow_module.IMAGE_SIZE,
            flow_module.PATCH_SIZE,
            flow_module.HIDDEN_DIM,
            flow_module.VIT_LAYERS,
            flow_module.VIT_HEADS,
        )
    ).to(device)

    normalizing_flow = flow_module.CondRealNVPFlow(
        dim=flow_module.HIDDEN_DIM,
        cond_dim=flow_module.HIDDEN_DIM,
        num_layers=flow_module.NF_LAYERS,
        hidden=flow_module.NF_HIDDEN,
    ).to(device)

    # Load weights
    student.load_state_dict(ckpt["student"], strict=False)
    teacher.load_state_dict(ckpt["teacher"], strict=False)
    normalizing_flow.load_state_dict(ckpt["nf"], strict=False)

    student.eval()
    teacher.eval()
    normalizing_flow.eval()

    # Get dataset
    dataset_module = _import_from(ROOT / "I-JEPA" / "i-jepa.py", "ijepa_eval_module")
    dataset = dataset_module.get_dataset("valid")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)

    # Sample masks
    grid = 8  # 8x8 grid
    ctx_mask_2d, tgt_mask_2d = dataset_module.sample_context_and_targets(grid)
    ctx_mask = ctx_mask_2d.view(-1).unsqueeze(0).expand(x.size(0), -1).to(device)
    tgt_mask = tgt_mask_2d.view(-1).unsqueeze(0).expand(x.size(0), -1).to(device)

    with torch.no_grad():
        teacher_tokens = teacher(x)

    # Extract target tokens
    idx = [tgt_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(x.size(0))]
    St = int(max([int(i.numel()) for i in idx]))
    D = teacher_tokens.size(2)
    teacher_tgt = torch.zeros((x.size(0), St, D), device=device, dtype=teacher_tokens.dtype)
    tgt_pad = torch.ones((x.size(0), St), dtype=torch.bool, device=device)

    for i, ids in enumerate(idx):
        L = int(ids.numel())
        if L == 0:
            continue
        teacher_tgt[i, :L] = teacher_tokens[i, ids]
        tgt_pad[i, :L] = False

    # Encode context
    ctx_tokens, ctx_pad = student.encode_context(x, ctx_mask)
    ctx_mean = ctx_tokens.mean(dim=1)  # Average context as conditioning

    # Generate samples using normalizing flow
    samples = []
    teacher_flat = teacher_tgt[~tgt_pad].cpu().numpy()
    num_target_tokens = teacher_flat.shape[0]

    # Standardize to exactly 10 tokens for fair comparison
    max_tokens = min(num_target_tokens, 10)  # Limit to 10 tokens max

    with torch.no_grad():
        # For each teacher token, generate draws_per_token samples
        for token_idx in range(max_tokens):
            # Use mean context as conditioning for this token
            ctx_single = ctx_mean[0:1]  # (1, D) - one context for all samples

            for _ in range(draws_per_token):
                # Start from single noise vector
                noise = torch.randn(1, D, device=device)  # (1, D)

                # Transform through normalizing flow
                transformed, _ = normalizing_flow(noise, ctx_single)  # (1, D)
                samples.append(transformed.squeeze(0))  # (D,)

    samples = torch.cat(samples, dim=0).cpu().numpy()

    # Ensure proper 2D shape
    if samples.ndim == 1:
        samples = samples.reshape(-1, D)  # [total_samples, D]
    elif samples.ndim == 3:
        samples = samples.reshape(-1, D)

    # Limit teacher tokens to match samples
    teacher_flat = teacher_flat[:max_tokens]

    return samples, teacher_flat


def main():
    print("Creating distribution visualizations for all models...")

    # Create visualizations for all 5 models
    models = []

    try:
        print("Sampling from MDN model...")
        mdn_samples, mdn_teacher = sample_mdn()
        models.append(("MDN", mdn_samples, mdn_teacher))
        print(f"Generated {mdn_samples.shape[0]} MDN samples from {mdn_teacher.shape[0]} teacher tokens")
    except Exception as e:
        print(f"Error sampling MDN model: {e}")

    try:
        print("Sampling from Autoregressive model...")
        arn_samples, arn_teacher = sample_autoregressive()
        models.append(("Autoregressive", arn_samples, arn_teacher))
        print(f"Generated {arn_samples.shape[0]} Autoregressive samples from {arn_teacher.shape[0]} teacher tokens")
    except Exception as e:
        print(f"Error sampling Autoregressive model: {e}")

    try:
        print("Sampling from Deterministic model...")
        base_samples, base_teacher = sample_baseline()
        models.append(("Deterministic", base_samples, base_teacher))
        print(f"Generated {base_samples.shape[0]} Deterministic samples from {base_teacher.shape[0]} teacher tokens")
    except Exception as e:
        print(f"Error sampling Deterministic model: {e}")

    try:
        print("Sampling from Diffusion model...")
        diff_samples, diff_teacher = sample_diffusion_simple()
        models.append(("Diffusion", diff_samples, diff_teacher))
        print(f"Generated {diff_samples.shape[0]} Diffusion samples from {diff_teacher.shape[0]} teacher tokens")
    except Exception as e:
        print(f"Error sampling Diffusion model: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("Sampling from Flow Matching model...")
        flow_samples, flow_teacher = sample_flow_matching_simple()
        models.append(("Flow Matching", flow_samples, flow_teacher))
        print(f"Generated {flow_samples.shape[0]} Flow Matching samples from {flow_teacher.shape[0]} teacher tokens")
    except Exception as e:
        print(f"Error sampling Flow Matching model: {e}")
        import traceback
        traceback.print_exc()

    # Create combined comparison figure
    if models:
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(len(models), 2)

        for idx, (name, samples, teacher) in enumerate(models):
            print(f"Creating visualization for {name}...")
            sample_2d, teacher_2d, _ = _project(samples, teacher, dims=2)

            ax2d = fig.add_subplot(gs[idx, 0])
            hb = _hexbin(ax2d, sample_2d, teacher_2d, f"{name} PCA-2D density")
            fig.colorbar(hb, ax=ax2d, fraction=0.046, pad=0.04)

            print(f"  - Teacher tokens: {teacher.shape[0]} cyan points")
            print(f"  - Generated samples: {samples.shape[0]} points in heatmap")

        plt.tight_layout()
        combined_path = FIG_DIR / "all_models_comparison.png"
        plt.savefig(combined_path, dpi=220)
        print(f"Saved combined comparison to {combined_path}")

        # Also save to root
        root_combined_path = ROOT / "all_models_distribution.png"
        plt.savefig(root_combined_path, dpi=220)
        print(f"Saved combined comparison to {root_combined_path}")

        # Individual visualizations
        for name, samples, teacher in models:
            sample_2d, teacher_2d, _ = _project(samples, teacher, dims=2)

            plt.figure(figsize=(8, 6))
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

            # Save individual files
            indiv_path = FIG_DIR / f"{name.lower().replace(' ', '_')}_distribution.png"
            plt.savefig(indiv_path, dpi=220)

            root_indiv_path = ROOT / f"{name.lower().replace(' ', '_')}_distribution.png"
            plt.savefig(root_indiv_path, dpi=220)

            print(f"Saved {name} visualization to {indiv_path} and {root_indiv_path}")
            plt.close()


if __name__ == "__main__":
    main()