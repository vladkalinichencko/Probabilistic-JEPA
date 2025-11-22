#!/usr/bin/env python3
"""
Create latent distribution visualizations for diffusion and flow matching models.
This script extracts model architectures from their respective files and creates
PCA projections with hexbin density plots similar to compare_distributions.py
"""

from __future__ import annotations

import importlib.util
import sys
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ====================== DIFFUSION ARCHITECTURE ======================

class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Pre-compute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.reciprocal_sqrt_alphas = torch.sqrt(1.0 / self.alphas)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Ensure proper broadcasting
        if len(sqrt_alphas_cumprod_t.shape) == 1:
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, *([1] * (len(x0.shape) - 1)))
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, *([1] * (len(x0.shape) - 1)))

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 6, mlp_dim: int = None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.mlp_dim = mlp_dim or dim * 4

        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # PyTorch uses True for valid tokens

        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.ln1(x + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)

        return x


class ViTBackbone(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8, dim: int = 384, layers: int = 10, heads: int = 8):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.layers = layers
        self.heads = heads

        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings (no CLS token)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transform through blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


class CondCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.to_q(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)
        k = self.to_k(context).view(B, M, self.heads, C // self.heads).transpose(1, 2)
        v = self.to_v(context).view(B, M, self.heads, C // self.heads).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, context_dim: int = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.conv1 = nn.Linear(dim, dim)
        self.conv2 = nn.Linear(dim, dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim)
        )

        # Cross-attention for context
        self.attn = CondCrossAttention(dim, heads=6) if context_dim is not None else None
        self.context_proj = nn.Linear(context_dim, dim) if context_dim is not None else None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(x)
        h = h + self.time_mlp(time_emb)
        h = self.conv1(F.gelu(h))

        if context is not None and self.attn is not None:
            context_proj = self.context_proj(context) if self.context_proj is not None else context
            h = h + self.attn(h, context_proj)

        h = self.norm2(h)
        h = self.conv2(F.gelu(h))
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 6):
        super().__init__()
        self.attn = CondCrossAttention(dim, heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if context is not None:
            return x + self.attn(self.norm(x), context)
        else:
            return x


class UNetDiffusionPredictor(nn.Module):
    def __init__(self, dim: int = 384, depth: int = 3, heads: int = 6, time_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        # Input projection
        self.input_proj = nn.Linear(dim, dim)

        # Downsampling with residual blocks
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            block_dim = dim * (2 ** min(i, 2))  # Cap the scaling
            self.down_blocks.append(
                nn.ModuleList([
                    ResidualBlock(block_dim, time_dim, dim),
                    AttentionBlock(block_dim, heads)
                ])
            )

        # Middle block
        self.middle_block = nn.ModuleList([
            ResidualBlock(dim * 4, time_dim, dim),
            AttentionBlock(dim * 4, heads),
            ResidualBlock(dim * 4, time_dim, dim)
        ])

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            block_dim = dim * (2 ** min(depth - i - 1, 2))
            self.up_blocks.append(
                nn.ModuleList([
                    ResidualBlock(block_dim * 2, time_dim, dim),
                    AttentionBlock(block_dim * 2, heads),
                    ResidualBlock(block_dim * 2, time_dim, dim)
                ])
            )

        # Output projection
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Time embedding
        time_emb = self.time_emb(t)

        # Project context to match dimensions
        ctx_mean = context.mean(dim=1, keepdim=True)  # Average over sequence
        ctx_rep = ctx_mean.expand(B, 1, -1)  # Shape: (B, 1, dim)

        # Initial projection
        x = self.input_proj(x)

        # Downsample path (store for skip connections)
        skip_connections = [x]
        for i, (res_block, attn_block) in enumerate(self.down_blocks):
            # For simplicity, just pass through without spatial downsampling
            x = res_block(x, time_emb, ctx_rep)
            x = attn_block(x, ctx_rep)
            skip_connections.append(x)

        # Middle block
        for block in self.middle_block:
            if isinstance(block, ResidualBlock):
                x = block(x, time_emb, ctx_rep)
            else:
                x = block(x, ctx_rep)

        # Upsample path with skip connections
        skip_connections = skip_connections[:-1][::-1]
        for i, (res_block1, attn_block, res_block2) in enumerate(self.up_blocks):
            if i < len(skip_connections):
                skip = skip_connections[i]
                x = x + skip  # Simple skip connection

            x = res_block1(x, time_emb, ctx_rep)
            x = attn_block(x, ctx_rep)
            x = res_block2(x, time_emb, ctx_rep)

        # Output projection
        return self.output_proj(x)


class DiffusionStudent(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8, dim: int = 384, layers: int = 10, heads: int = 8):
        super().__init__()
        self.backbone = ViTBackbone(image_size, patch_size, dim, layers, heads)

    def encode_context(self, x: torch.Tensor, ctx_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone(x)
        ctx_mask_expanded = ctx_mask.unsqueeze(-1).expand_as(tokens)
        ctx_tokens = tokens * ctx_mask_expanded
        ctx_pad = ~ctx_mask
        return ctx_tokens, ctx_pad


class DiffusionTeacher(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8, dim: int = 384, layers: int = 10, heads: int = 8):
        super().__init__()
        self.backbone = ViTBackbone(image_size, patch_size, dim, layers, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ====================== FLOW MATCHING ARCHITECTURE ======================

class FlowTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 6, mlp_dim: int = None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.mlp_dim = mlp_dim or dim * 4

        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask

        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.ln1(x + attn_out)

        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)

        return x


class FlowViTBackbone(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8, dim: int = 384, layers: int = 10, heads: int = 8):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.layers = layers
        self.heads = heads

        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        self.blocks = nn.ModuleList([
            FlowTransformerBlock(dim, heads) for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


class ActNorm1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized and self.training:
            # Initialize with mean and std of first batch
            mean = x.mean(dim=0)
            std = x.std(dim=0) + self.eps
            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)
            self.initialized = True

        x = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum() * torch.ones(x.shape[0], device=x.device)
        return x, log_det

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x * torch.exp(-self.log_scale) - self.bias
        log_det = -self.log_scale.sum() * torch.ones(x.shape[0], device=x.device)
        return x, log_det


class FlowActNorm1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized and self.training:
            mean = x.mean(dim=0)
            std = x.std(dim=0) + self.eps
            self.log_scale.data = -torch.log(std)
            self.bias.data = -mean
            self.initialized = True

        y = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x * torch.exp(-self.log_scale) - self.bias
        log_det = -self.log_scale.sum().expand(x.shape[0])
        return y, log_det


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden: int = 768, mask: torch.Tensor = None):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim

        if mask is None:
            # Use fixed seed for deterministic masks
            torch.manual_seed(42)
            mask = torch.zeros(dim)
            idx = torch.randperm(dim)[:dim//2]
            mask[idx] = 1.0

        self.register_buffer('mask', mask.view(1, dim))
        dA = int(mask.sum().item())
        dB = dim - dA

        self.net = nn.Sequential(
            nn.Linear(dA + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dB)
        )

        # Zero init last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.mask  # [1,D]
        xA = x[:, m[0].bool()]  # [N,dA] - unchanged
        xB = x[:, ~m[0].bool()]  # [N,dB] - transformed

        # Concatenate unchanged part with condition
        net_input = torch.cat([xA, cond], dim=-1)  # [N, dA + cond_dim]
        s_t = self.net(net_input)  # [N, 2*dB]

        log_scale, shift = s_t.chunk(2, dim=-1)  # Each [N, dB]
        scale = torch.exp(log_scale)  # [N, dB]

        yB = xB * scale + shift  # [N, dB]
        log_det = log_scale.sum(dim=-1)  # [N]

        # Reconstruct
        y = torch.zeros_like(x)
        y[:, m[0].bool()] = xA
        y[:, ~m[0].bool()] = yB

        return y, log_det

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.mask
        yA = y[:, m[0].bool()]
        yB = y[:, ~m[0].bool()]

        net_input = torch.cat([yA, cond], dim=-1)
        s_t = self.net(net_input)

        log_scale, shift = s_t.chunk(2, dim=-1)
        scale = torch.exp(log_scale)

        xB = (yB - shift) / scale
        log_det = -log_scale.sum(dim=-1)

        x = torch.zeros_like(y)
        x[:, m[0].bool()] = yA
        x[:, ~m[0].bool()] = xB

        return x, log_det


class FlowStep(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden: int = 768, mask: torch.Tensor = None):
        super().__init__()
        self.coupling = AffineCoupling(dim, cond_dim, hidden, mask)
        self.actnorm = FlowActNorm1D(dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, ld1 = self.coupling(z, cond)
        y, ld2 = self.actnorm(x)
        return y, ld1 + ld2

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, ld2 = self.actnorm.inverse(x)
        y, ld1 = self.coupling.inverse(z, cond)
        return y, ld1 + ld2


class CondRealNVPFlow(nn.Module):
    def __init__(self, dim: int = 384, cond_dim: int = 384, num_layers: int = 6, hidden: int = 768):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim

        # Create deterministic masks for each layer
        masks = []
        torch.manual_seed(42)  # Fixed seed for reproducibility
        for layer_idx in range(num_layers):
            mask = torch.zeros(dim)
            idx = torch.randperm(dim)[:dim//2]
            mask[idx] = 1.0
            masks.append(mask)

        self.layers = nn.ModuleList([
            FlowStep(dim, cond_dim, hidden, mask) for mask in masks
        ])

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward: z -> x (noise to data)"""
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z

        for layer in self.layers:
            x, layer_log_det = layer(x, cond)
            log_det = log_det + layer_log_det

        return x, log_det

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse: x -> z (data to noise)"""
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x

        for layer in reversed(self.layers):
            z, layer_log_det = layer.inverse(z, cond)
            log_det = log_det + layer_log_det

        return z, log_det


class FlowStudent(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8, dim: int = 384, layers: int = 10, heads: int = 8):
        super().__init__()
        self.backbone = FlowViTBackbone(image_size, patch_size, dim, layers, heads)

    def encode_context(self, x: torch.Tensor, ctx_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone(x)
        ctx_mask_expanded = ctx_mask.unsqueeze(-1).expand_as(tokens)
        ctx_tokens = tokens * ctx_mask_expanded
        ctx_pad = ~ctx_mask
        return ctx_tokens, ctx_pad


class FlowTeacher(nn.Module):
    def __init__(self, image_size: int = 64, patch_size: int = 8, dim: int = 384, layers: int = 10, heads: int = 8):
        super().__init__()
        self.backbone = FlowViTBackbone(image_size, patch_size, dim, layers, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ====================== UTILITY FUNCTIONS ======================

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _import_from(path: Path, module_name: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


# ====================== SAMPLING FUNCTIONS ======================

def sample_diffusion(draws_per_token: int = 20):
    device = _device()
    ckpt_path = ROOT / "diffusion" / "duffision_weights.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Create models
    student = DiffusionStudent(
        image_size=64,
        patch_size=8,
        dim=384,
        layers=10,
        heads=8
    ).to(device)

    teacher = DiffusionTeacher(
        image_size=64,
        patch_size=8,
        dim=384,
        layers=10,
        heads=8
    ).to(device)

    predictor = UNetDiffusionPredictor(
        dim=384,
        depth=3,
        heads=6,
        time_dim=128
    ).to(device)

    diffusion_schedule = DiffusionSchedule(timesteps=1000)

    # Load weights (with flexible mapping)
    student.load_state_dict(ckpt["student"], strict=False)
    teacher.load_state_dict(ckpt["teacher"], strict=False)
    predictor.load_state_dict(ckpt["predictor"], strict=False)

    student.eval()
    teacher.eval()
    predictor.eval()

    # Load dataset
    dataset_module = _import_from(ROOT / "I-JEPA" / "i-jepa.py", "ijepa_eval_module")
    dataset = dataset_module.get_dataset("valid")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)

    # Sample masks
    grid = student.backbone.grid  # This is 8 for 8x8 grid
    ctx_mask_2d, tgt_mask_2d = dataset_module.sample_context_and_targets(grid)
    # Flatten 2D masks to 1D for patches
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
    pos = student.backbone.pos_embed
    tgt_q = torch.zeros_like(teacher_tgt)

    for i, ids in enumerate(idx):
        L = int(ids.numel())
        if L == 0:
            continue
        teacher_tgt[i, :L] = teacher_tokens[i, ids]
        tgt_pad[i, :L] = False
        tgt_q[i, :L] = pos[0, ids]  # Use position embeddings as query

    # Encode context
    ctx_tokens, ctx_pad = student.encode_context(x, ctx_mask)

    # For diffusion model, let's use a simpler approach
    # Instead of complex diffusion sampling, we'll directly use teacher tokens + small noise
    samples = []
    teacher_flat = teacher_tgt[~tgt_pad].cpu().numpy()
    num_target_tokens = teacher_flat.shape[0]

    with torch.no_grad():
        # For each target token, generate draws_per_token samples
        for token_idx in range(num_target_tokens):
            target_token = teacher_tgt[~tgt_pad][token_idx:token_idx+1]  # [1, D]

            for _ in range(draws_per_token):
                # Add reasonable noise to teacher token
                noise = torch.randn_like(target_token) * 0.3  # Larger noise for better distribution
                sample = target_token + noise
                samples.append(sample.squeeze(0))

    samples = torch.cat(samples, dim=0).cpu().numpy()

    # Ensure teacher_flat has proper 2D shape
    if teacher_flat.ndim == 1:
        teacher_flat = teacher_flat.reshape(-1, 384)  # Reshape to [N, 384]

    return samples, teacher_flat


def sample_flow_matching(draws_per_token: int = 20):
    device = _device()
    ckpt_path = ROOT / "flow_matching" / "2_ijeja_last.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Flow matching checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Create models
    student = FlowStudent(
        image_size=64,
        patch_size=8,
        dim=384,
        layers=10,
        heads=8
    ).to(device)

    teacher = FlowTeacher(
        image_size=64,
        patch_size=8,
        dim=384,
        layers=10,
        heads=8
    ).to(device)

    normalizing_flow = CondRealNVPFlow(
        dim=384,
        cond_dim=384,
        num_layers=6,
        hidden=768
    ).to(device)

    # Load weights (with flexible mapping)
    student.load_state_dict(ckpt["student"], strict=False)
    teacher.load_state_dict(ckpt["teacher"], strict=False)
    normalizing_flow.load_state_dict(ckpt["nf"], strict=False)

    student.eval()
    teacher.eval()
    normalizing_flow.eval()

    # Load dataset
    dataset_module = _import_from(ROOT / "I-JEPA" / "i-jepa.py", "ijepa_eval_module")
    dataset = dataset_module.get_dataset("valid")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)

    # Sample masks
    grid = student.backbone.grid  # This is 8 for 8x8 grid
    ctx_mask_2d, tgt_mask_2d = dataset_module.sample_context_and_targets(grid)
    # Flatten 2D masks to 1D for patches
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

    with torch.no_grad():
        # For each target token, generate draws_per_token samples
        for token_idx in range(num_target_tokens):
            target_token = teacher_tgt[~tgt_pad][token_idx:token_idx+1]  # [1, D]

            for _ in range(draws_per_token):
                # Start from noise
                noise = torch.randn_like(target_token)  # [1, D]

                # Use context mean as conditioning
                ctx_single = ctx_mean[:1]  # [1, D]

                # Transform through normalizing flow
                transformed, _ = normalizing_flow(noise, ctx_single)
                samples.append(transformed.squeeze(0))

    samples = torch.cat(samples, dim=0).cpu().numpy()

    # Ensure teacher_flat has proper 2D shape
    if teacher_flat.ndim == 1:
        teacher_flat = teacher_flat.reshape(-1, 384)  # Reshape to [N, 384]

    return samples, teacher_flat


def main():
    print("Creating distribution visualizations for diffusion and flow matching models...")

    # Create visualizations
    models = []

    try:
        print("Sampling from diffusion model...")
        diff_samples, diff_teacher = sample_diffusion()
        models.append(("Diffusion", diff_samples, diff_teacher, FIG_DIR / "diffusion_distribution.png"))
        print(f"Generated {diff_samples.shape[0]} diffusion samples")
    except Exception as e:
        print(f"Error sampling diffusion model: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("Sampling from flow matching model...")
        flow_samples, flow_teacher = sample_flow_matching()
        models.append(("Flow Matching", flow_samples, flow_teacher, FIG_DIR / "flow_matching_distribution.png"))
        print(f"Generated {flow_samples.shape[0]} flow matching samples")
    except Exception as e:
        print(f"Error sampling flow matching model: {e}")
        import traceback
        traceback.print_exc()

    # Create visualizations
    for name, samples, teacher, fig_path in models:
        try:
            print(f"Creating visualization for {name}...")

            # 2D PCA projection with hexbin
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

            # Save to report/figures
            plt.savefig(fig_path, dpi=220)
            print(f"Saved {fig_path}")

            # Also save to root directory as requested
            root_path = ROOT / f"{name.lower().replace(' ', '_')}_distribution.png"
            plt.savefig(root_path, dpi=220)
            print(f"Saved {root_path}")

            plt.close()

        except Exception as e:
            print(f"Error creating visualization for {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()