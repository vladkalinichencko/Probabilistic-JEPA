"""
Visualization suite for i-JEPA + Multivariate MDN

Run with:
    python visualize_mdn.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from MDN.mdn import (
    ViTBackbone, Student, Teacher, MDNPredictor,
    get_dataset, sample_context_and_targets, MDN_K,
    HIDDEN_DIM, IMAGE_SIZE, PATCH_SIZE, VIT_LAYERS, VIT_HEADS,
    LAST_CKPT, MIN_CTX_PATCHES
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./mdn_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# Load model from checkpoint
# ------------------------------
def load_model():
    print("Loading checkpoint:", LAST_CKPT)
    ckpt = torch.load(LAST_CKPT, map_location=DEVICE)

    student = Student(ViTBackbone(
        IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS
    )).to(DEVICE)

    teacher = Teacher(ViTBackbone(
        IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS
    )).to(DEVICE)

    predictor = MDNPredictor(
        HIDDEN_DIM, num_layers=2, num_heads=VIT_HEADS, K=MDN_K
    ).to(DEVICE)

    student.load_state_dict(ckpt['student'])
    teacher.load_state_dict(ckpt['teacher'])
    predictor.load_state_dict(ckpt['predictor'])

    student.eval(); teacher.eval(); predictor.eval()

    return student, teacher, predictor

# ------------------------------
# Simple batch iterator
# ------------------------------
def get_batch(n=1):
    ds = get_dataset("valid")
    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=True)
    for x, _ in loader:
        return x.to(DEVICE)

# ------------------------------
# MDN parameter analysis
# ------------------------------
def visualize_mixture_weights(pi):
    """
    pi: [B, St, K]
    """
    flat = pi.reshape(-1, pi.size(-1)).detach().cpu().numpy()

    plt.figure(figsize=(6,4))
    plt.hist(flat.flatten(), bins=50)
    plt.title("Histogram: Mixture Weights π")
    plt.xlabel("π values")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/mixture_weights.png")
    plt.close()

def visualize_sigma(sigma):
    """
    sigma: [B, St, K, D]
    """
    flat = sigma.detach().cpu().numpy().flatten()

    plt.figure(figsize=(6,4))
    plt.hist(flat, bins=50)
    plt.title("Distribution of σ (stddevs)")
    plt.xlabel("σ")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/sigma_distribution.png")
    plt.close()

def visualize_entropy(pi):
    """
    Entropy of each mixture distribution: H(pi)
    """
    p = pi.detach().cpu().numpy()
    eps = 1e-12
    H = -np.sum(p * np.log(p + eps), axis=-1).flatten()

    plt.figure(figsize=(6,4))
    plt.hist(H, bins=40)
    plt.title("Entropy of Mixture Weights H(π)")
    plt.xlabel("entropy")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/entropy.png")
    plt.close()

# ------------------------------
# PCA projection utilities
# ------------------------------
def pca_project(X):
    """
    X: [N, D]
    """
    X = X.detach().cpu().numpy()
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

# ------------------------------
# Visualization of μ and samples
# ------------------------------
def visualize_embeddings(mu, samples, teacher_targets):
    """
    PCA visualization of:
      - teacher target embeddings
      - MDN component means
      - random samples from mixture
    """
    B, St, K, D = mu.shape
    mu_flat = mu[0].reshape(St*K, D)
    samples_flat = samples[0]
    target_flat = teacher_targets[0]

    points = torch.cat([mu_flat, samples_flat, target_flat], dim=0)
    proj = pca_project(points)

    N_mu = mu_flat.size(0)
    N_samples = samples_flat.size(0)

    plt.figure(figsize=(7,7))
    plt.scatter(proj[:N_mu,0], proj[:N_mu,1], s=8, label="MDN means μ_k")
    plt.scatter(proj[N_mu:N_mu+N_samples,0],
                proj[N_mu:N_mu+N_samples,1], s=8, alpha=0.5, label="Samples")
    plt.scatter(proj[N_mu+N_samples:,0],
                proj[N_mu+N_samples:,1], s=20, label="Teacher targets")

    plt.legend()
    plt.title("PCA: μ_k, Samples, Teacher Targets")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/pca_means_samples_targets.png")
    plt.close()

# ------------------------------
# Main visualization pipeline
# ------------------------------
def main():
    student, teacher, predictor = load_model()

    print("Sampling batch…")
    x = get_batch(1)
    B = x.size(0)
    grid = student.backbone.grid
    T = grid*grid

    # Sample mask
    ctx_mask, tgt_mask = sample_context_and_targets(grid)
    ctx_mask = ctx_mask.unsqueeze(0).to(DEVICE)
    tgt_mask = tgt_mask.unsqueeze(0).to(DEVICE)

    # Teacher target tokens
    with torch.no_grad():
        t_tokens = teacher(x)                      # [B,T,D]
    idx = tgt_mask[0].nonzero(as_tuple=False).squeeze(1)
    St = idx.numel()
    D = t_tokens.size(-1)

    teacher_tgt = t_tokens[:, idx, :]             # [B,St,D]
    tgt_pad = torch.zeros((B, St), dtype=torch.bool, device=DEVICE)

    # Build target queries
    pos = student.backbone.pos_embed
    tgt_q = predictor.q_token[None, :].expand(St, -1) + pos[0, idx]
    tgt_q = tgt_q.unsqueeze(0)                    # [B,St,D]

    # Context encoding
    ctx_tok, ctx_pad = student.encode_context(x, ctx_mask)

    # MDN outputs
    with torch.no_grad():
        pi, mu, sigma = predictor(ctx_tok, ctx_pad, tgt_q, tgt_pad)

    # Visualize mixture parameters
    visualize_mixture_weights(pi)
    visualize_sigma(sigma)
    visualize_entropy(pi)

    # Draw samples from mixture for PCA
    with torch.no_grad():
        # [B,St,K,D]
        comp = torch.distributions.Normal(mu, sigma)
        # sample component per token
        cat = torch.distributions.Categorical(pi)
        comp_idx = cat.sample().unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,D)
        chosen_mu = mu.gather(2, comp_idx).squeeze(2)
        chosen_sigma = sigma.gather(2, comp_idx).squeeze(2)
        sample = torch.normal(chosen_mu, chosen_sigma)

    visualize_embeddings(mu, sample, teacher_tgt)

    print("Saved all visualizations to:", SAVE_DIR)

if __name__ == "__main__":
    main()
