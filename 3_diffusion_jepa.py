import os
import math
import random
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import subprocess
import time
import socket
import itertools
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from visualization import (
    ModelInterpreter,
    compute_diag_metrics,
    tb_log_scalars,
    tb_log_mask_overlay,
    tb_log_vector_projection,
    FixedProjector2D,
    tb_log_knn_convergence,
)

# ------------------
# Constants / Defaults
# ------------------
NUM_CLASSES = 100
IMAGE_SIZE = 64
EPOCHS = 400
BATCH_SIZE = 600
LR = 5e-5
WEIGHT_DECAY = 1e-4
DEVICE_TYPE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
DEVICE = torch.device(DEVICE_TYPE)

# Logging / runtime config
RUN_ID = time.strftime("%Y%m%d-%H%M%S")

# DataLoader parameters
TRAIN_NUM_WORKERS = 2
VAL_NUM_WORKERS = 2
BANK_NUM_WORKERS = 2
PIN_MEMORY = DEVICE_TYPE == "cuda"

# Diagnostics cadence
LOG_INTERVAL_MULTIPLIER = 2
VAL_INTERVAL_MULTIPLIER = 200

# Optimizer aux
GRAD_CLIP_MAX_NORM = 3.0

# ViT / JEPA sizes
PATCH_SIZE = 8
HIDDEN_DIM = 384
VIT_LAYERS = 10
VIT_HEADS = 8

# ------------------
# Diffusion config
# ------------------
DIFFUSION_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
UNET_DEPTH = 3
UNET_HIDDEN_MULT = 2
UNET_NUM_HEADS = 6
TIME_EMB_DIM = 128
INFERENCE_TIMESTEPS = 50  # For DDIM-style fast sampling

# Masking / targets
NUM_TARGET_BLOCKS = 4
TARGET_SCALE = (0.15, 0.20)
TARGET_AR = (0.75, 1.50)
MIN_CTX_PATCHES = 8

# EMA
MOMENTUM_INIT = 0.998
MOMENTUM_FINAL = 1.0

CHECKPOINT_DIR = os.path.abspath("./checkpoints")
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "3_diffusion_last.pt")
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "3_diffusion_best_knn.pt")

TB_LOGDIR = os.path.abspath("./runs")
TB_PORT = 43802
TB_SUFFIX = "tb-sage"
TB_FLUSH_SECS = 5

# Evaluation / diagnostics
KNN_K = 20
VAL_LOSS_MAX_BATCHES = 5
PROJECTOR_PERCENTILE = 99.0


def _proxy_url(port: int, suffix: str, absolute: bool = True) -> str:
    rel = f"/proxy/absolute/{port}/{suffix}/"
    if not absolute:
        return rel
    root = (
        os.environ.get("JUPYTERHUB_ROOT_URL")
        or os.environ.get("NB_URL")
        or os.environ.get("NB_HOST")
        or os.environ.get("AIM_UI_ORIGIN")
        or "https://45957f89c897.innodatahub.innopolis.university"
    )
    root = root.rstrip("/")
    return f"{root}{rel}"


def start_tb_ui(
    logdir: str = os.path.abspath("./runs"), port: int = 43802, suffix: str = "tb-sage"
) -> Optional[str]:
    try:
        from IPython.display import HTML, display

        IN_NOTEBOOK = "ipykernel" in sys.modules
    except Exception:
        HTML = None
        display = None
        IN_NOTEBOOK = False

    os.makedirs(logdir, exist_ok=True)
    try:
        subprocess.run(
            ["bash", "-lc", f"pkill -f 'tensorboard .*{port}' || true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    prefix = f"/proxy/absolute/{port}/{suffix}"
    try:
        proc = subprocess.Popen(
            [
                "tensorboard",
                "--logdir",
                logdir,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--path_prefix",
                prefix,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        print("tensorboard не найден. Установите пакет 'tensorboard'.")
        return None

    def port_open(p):
        s = socket.socket()
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", p))
            s.close()
            return True
        except OSError:
            return False

    for _ in range(60):
        if port_open(port):
            break
        time.sleep(0.5)

    if not port_open(port):
        try:
            lines = list(itertools.islice(proc.stdout, 200))
            print("TensorBoard не поднялся. Логи:\n" + "".join(lines[-200:]))
        except Exception:
            print("TensorBoard не поднялся и логи недоступны.")
        return None

    rel_url = _proxy_url(port, suffix, absolute=False)
    abs_url = _proxy_url(port, suffix, absolute=True)
    if IN_NOTEBOOK and HTML and display:
        try:
            display(
                HTML(f'<a href="{abs_url}" target="_blank">открыть TensorBoard</a>')
            )
        except Exception:
            pass
    print(f"TensorBoard: {abs_url} (logdir={logdir})")
    print("(относительный в Jupyter)", rel_url)
    return abs_url


# --------------
# Dataset & TFMs
# --------------
TRAIN_MEAN_STD = None


class ToRGB:
    def __call__(self, img):
        return img.convert("RGB")


class TinyImageNetTorch(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]["image"]
        label = self.data[index]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def _filter_label_lt_num_classes(example):
    return int(example.get("label", -1)) < NUM_CLASSES


def get_dataset(split: str) -> TinyImageNetTorch:
    def load_tinyimagenet(_split):
        dataset = load_dataset("zh-plus/tiny-imagenet", split=_split)
        dataset = dataset.filter(_filter_label_lt_num_classes)
        return dataset

    def compute_statistics(dataset):
        n_channels = 3
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        n_samples = len(dataset)
        for sample in dataset:
            img = sample["image"]
            img = transforms.ToTensor()(img)
            mean += img.mean(dim=(1, 2))
            std += img.std(dim=(1, 2))
        mean /= n_samples
        std /= n_samples
        return mean.numpy(), std.numpy()

    global TRAIN_MEAN_STD
    if TRAIN_MEAN_STD is None:
        train_raw = load_tinyimagenet("train")
        TRAIN_MEAN_STD = compute_statistics(train_raw)
    mean, std = TRAIN_MEAN_STD

    raw_dataset = load_tinyimagenet(split)

    if split == "train":
        tfms = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
            ]
        )
    else:
        tfms = transforms.Compose(
            [
                ToRGB(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
            ]
        )
    return TinyImageNetTorch(raw_dataset, transform=tfms)


# --------------
# ViT backbone
# --------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch_embed(x)
        t = t.flatten(2).transpose(1, 2)
        t = t + self.pos_embed
        for blk in self.transformer_layers:
            t = blk(t, key_padding_mask=None)
        return self.norm(t)


# -----------------
# Student / Teacher
# -----------------
class Student(nn.Module):
    def __init__(self, backbone: ViTBackbone):
        super().__init__()
        self.backbone = backbone

    def encode_context(
        self, x: torch.Tensor, context_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            B = x.size(0)
        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        pos = self.backbone.pos_embed
        ctx_lists: List[torch.Tensor] = []
        lengths: List[int] = []
        for i in range(tokens.size(0)):
            idx = context_mask[i].nonzero(as_tuple=False).squeeze(1)
            ti = tokens[i, idx]
            pi = pos[0, idx]
            ctx_lists.append(ti + pi)
            lengths.append(int(idx.numel()))
        Sc = int(max(lengths)) if lengths else 0
        D = tokens.size(2)
        ctx_pad = torch.ones((B, Sc), dtype=torch.bool, device=tokens.device)
        ctx_out = torch.zeros((B, Sc, D), dtype=tokens.dtype, device=tokens.device)
        for i, ci in enumerate(ctx_lists):
            L = ci.size(0)
            ctx_out[i, :L] = ci
            ctx_pad[i, :L] = False
        for blk in self.backbone.transformer_layers:
            ctx_out = blk(ctx_out, key_padding_mask=ctx_pad)
        ctx_out = self.backbone.norm(ctx_out)
        return ctx_out, ctx_pad


class Teacher(nn.Module):
    def __init__(self, backbone: ViTBackbone):
        super().__init__()
        self.backbone = backbone
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def ema_update(self, student: Student, m: float):
        for p_t, p_s in zip(self.backbone.parameters(), student.backbone.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))


# --------------------------------------------
# Diffusion Components
# --------------------------------------------
class DiffusionSchedule:
    """DDPM noise schedule with helpers for forward/reverse diffusion."""

    def __init__(
        self,
        timesteps: int = DIFFUSION_TIMESTEPS,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
    ):
        self.timesteps = timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, dtype=torch.float32
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CondCrossAttention(nn.Module):
    """
    Per-target conditioning via cross-attention:
    query  = target positional embeddings (plus learned target token)
    key/val = context tokens
    Output: [B, St, D] per target, aligned with target positions.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.tgt_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        ctx_pad: torch.Tensor,
        tgt_pos: torch.Tensor,
    ) -> torch.Tensor:
        # ctx_tokens: [B, Sc, D], ctx_pad: [B, Sc] (True=pad), tgt_pos: [B, St, D]
        B, St, D = tgt_pos.size()
        # broadcast learned target token to [B, St, D]
        tgt_tok = self.tgt_token.expand(B, St, D)
        query = tgt_pos + tgt_tok  # [B, St, D]
        # MultiheadAttention with batch_first=True
        # key_padding_mask: [B, Sc], True = ignore
        h, _ = self.attn(
            query, ctx_tokens, ctx_tokens, key_padding_mask=ctx_pad
        )  # [B, St, D]
        h = self.ln(h)
        return h


class ResidualBlock(nn.Module):
    """Residual block with time and context conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(cond_dim, dim * 2))
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(cond)
        scale, shift = time_emb.chunk(2, dim=-1)
        h = self.block1(x)
        h = h * (1 + scale) + shift
        h = self.block2(h)
        return x + h


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h, _ = self.attn(h, h, h)
        return x + h


class UNetDiffusionPredictor(nn.Module):
    """U-Net-like architecture for noise prediction in diffusion JEPA.

    Predicts noise epsilon given:
    - Noised embedding h(t) = sqrt(alpha_t) * h_clean + sqrt(1-alpha_t) * epsilon
    - Context conditioning (pooled context tokens + positional encoding)
    - Timestep t
    """

    def __init__(
        self,
        dim: int = HIDDEN_DIM,
        time_emb_dim: int = TIME_EMB_DIM,
        depth: int = UNET_DEPTH,
        hidden_mult: int = UNET_HIDDEN_MULT,
        num_heads: int = UNET_NUM_HEADS,
    ):
        super().__init__()
        self.dim = dim
        self.time_emb_dim = time_emb_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        cond_dim = time_emb_dim + dim

        self.input_proj = nn.Linear(dim, dim)

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        dims = [dim]
        for i in range(depth):
            is_last = i == depth - 1
            out_dim = (
                dim * (hidden_mult ** (i + 1))
                if not is_last
                else dim * (hidden_mult**i)
            )

            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dims[-1], cond_dim),
                        ResidualBlock(dims[-1], cond_dim),
                        AttentionBlock(dims[-1], num_heads),
                    ]
                )
            )

            if not is_last:
                self.downsamples.append(nn.Linear(dims[-1], out_dim))
                dims.append(out_dim)

        self.mid_block1 = ResidualBlock(dims[-1], cond_dim)
        self.mid_attn = AttentionBlock(dims[-1], num_heads)
        self.mid_block2 = ResidualBlock(dims[-1], cond_dim)

        for i in reversed(range(depth)):
            is_last = i == 0
            in_dim = dims[i + 1] if i < len(dims) - 1 else dims[-1]
            out_dim = dims[i]

            self.upsamples.append(nn.Linear(in_dim, out_dim))

            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(out_dim * 2, cond_dim),
                        ResidualBlock(out_dim * 2, cond_dim),
                        AttentionBlock(out_dim * 2, num_heads),
                    ]
                )
            )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Noised embeddings [N, D]
            t: Timesteps [N]
            ctx_cond: Context conditioning [N, D] (pooled context + positional encoding)
        Returns:
            Predicted noise [N, D]
        """
        t_emb = self.time_embed(t)
        cond = torch.cat([t_emb, ctx_cond], dim=-1)

        h = self.input_proj(x).unsqueeze(1)

        skip_connections = []
        for res1, res2, attn in self.encoder_blocks:
            h_flat = h.squeeze(1)
            h = res1(h_flat, cond).unsqueeze(1)
            h_flat = h.squeeze(1)
            h = res2(h_flat, cond).unsqueeze(1)
            h = attn(h)
            skip_connections.append(h)
            if len(skip_connections) < len(self.encoder_blocks):
                h_flat = h.squeeze(1)
                h = self.downsamples[len(skip_connections) - 1](h_flat).unsqueeze(1)

        h_flat = h.squeeze(1)
        h = self.mid_block1(h_flat, cond).unsqueeze(1)
        h = self.mid_attn(h)
        h_flat = h.squeeze(1)
        h = self.mid_block2(h_flat, cond).unsqueeze(1)

        for i, (res1, res2, attn) in enumerate(self.decoder_blocks):
            if i > 0:
                h_flat = h.squeeze(1)
                h = self.upsamples[i - 1](h_flat).unsqueeze(1)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=-1)
            h_flat = h.squeeze(1)
            h = res1(h_flat, cond).unsqueeze(1)
            h_flat = h.squeeze(1)
            h = res2(h_flat, cond).unsqueeze(1)
            h = attn(h)

        h = self.output_proj(h.squeeze(1))
        return h


# --------------
# Mask utilities
# --------------
def _sample_blocks(
    grid: int, n_blocks: int, scale: Tuple[float, float], ar: Tuple[float, float]
) -> List[Tuple[int, int, int, int]]:
    blocks = []
    for _ in range(n_blocks):
        area = np.random.uniform(scale[0], scale[1]) * (grid * grid)
        r = np.random.uniform(ar[0], ar[1])
        h = int(round(np.sqrt(area / r)))
        w = int(round(h * r))
        h = max(1, min(h, grid))
        w = max(1, min(w, grid))
        y0 = np.random.randint(0, max(1, grid - h + 1))
        x0 = np.random.randint(0, max(1, grid - w + 1))
        blocks.append((y0, x0, h, w))
    return blocks


def build_mask_from_blocks(
    grid: int, blocks: List[Tuple[int, int, int, int]]
) -> torch.Tensor:
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    for y0, x0, h, w in blocks:
        mask[y0 : y0 + h, x0 : x0 + w] = True
    return mask.flatten()


def sample_context_and_targets(
    grid: int, min_ctx: int = MIN_CTX_PATCHES, max_tries: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    for _ in range(max_tries):
        ctx_block = _sample_blocks(grid, 1, (0.85, 1.0), (0.75, 1.5))[0]
        tgt_blocks = _sample_blocks(grid, NUM_TARGET_BLOCKS, TARGET_SCALE, TARGET_AR)
        ctx = torch.zeros(grid, grid, dtype=torch.bool)
        y0, x0, h, w = ctx_block
        ctx[y0 : y0 + h, x0 : x0 + w] = True
        tgt = torch.zeros(grid, grid, dtype=torch.bool)
        for yy, xx, hh, ww in tgt_blocks:
            tgt[yy : yy + hh, xx : xx + ww] = True
        ctx[tgt] = False
        if int(ctx.sum().item()) >= min_ctx and int(tgt.sum().item()) > 0:
            return ctx.flatten(), tgt.flatten()
    ctx = torch.ones(grid, grid, dtype=torch.bool)
    tgt_blocks = _sample_blocks(
        grid, max(1, NUM_TARGET_BLOCKS // 2), (0.10, 0.15), TARGET_AR
    )
    tgt = torch.zeros(grid, grid, dtype=torch.bool)
    for yy, xx, hh, ww in tgt_blocks:
        tgt[yy : yy + hh, xx : xx + ww] = True
    ctx[tgt] = False
    if int(ctx.sum().item()) == 0:
        ctx[0, 0] = True
        tgt[0, 0] = False
    return ctx.flatten(), tgt.flatten()


# ----------------
# Loss & training
# ----------------
def momentum_schedule(
    it: int, total_its: int, m0: float = MOMENTUM_INIT, m1: float = MOMENTUM_FINAL
) -> float:
    if total_its <= 1:
        return m1
    t = min(1.0, max(0.0, it / (total_its - 1)))
    return m1 - (m1 - m0) * (0.5 * (1.0 + math.cos(math.pi * t)))


@torch.no_grad()
def extract_global_embeddings(
    backbone: ViTBackbone, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        t = backbone(x)
        g = t.mean(dim=1)
        feats.append(g.cpu())
        labels.append(y.clone().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def knn_eval(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    k: int = KNN_K,
) -> float:
    train_norm = F.normalize(train_feats, dim=1)
    val_norm = F.normalize(val_feats, dim=1)
    sim = val_norm @ train_norm.t()
    topk = sim.topk(k, dim=1).indices
    neighbors = train_labels[topk]
    preds = torch.mode(neighbors, dim=1).values
    acc = (preds == val_labels).float().mean().item()
    return float(acc)


@torch.no_grad()
def ddim_sample(
    diffusion: DiffusionSchedule,
    predictor: UNetDiffusionPredictor,
    ctx_cond: torch.Tensor,
    num_steps: int = INFERENCE_TIMESTEPS,
) -> torch.Tensor:
    """DDIM sampling for faster inference."""
    device = ctx_cond.device
    N = ctx_cond.size(0)
    D = predictor.dim

    x = torch.randn(N, D, device=device)

    timesteps = torch.linspace(
        diffusion.timesteps - 1, 0, num_steps, dtype=torch.long, device=device
    )

    for i, t in enumerate(timesteps):
        t_batch = torch.full((N,), t, device=device, dtype=torch.long)

        predicted_noise = predictor(x, t_batch, ctx_cond)

        alpha_t = diffusion.alphas_cumprod[t]
        alpha_t_prev = (
            diffusion.alphas_cumprod[timesteps[i + 1]]
            if i < len(timesteps) - 1
            else torch.tensor(1.0, device=device)
        )

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        x_0_pred = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)

        x = sqrt_alpha_t_prev * x_0_pred + sqrt_one_minus_alpha_t_prev * predicted_noise

    return x


def train_run():
    tb_url = start_tb_ui(logdir=os.path.abspath("./runs"), port=43802, suffix="tb-sage")
    writer = SummaryWriter(
        log_dir=os.path.join(os.path.abspath("./runs"), RUN_ID),
        flush_secs=TB_FLUSH_SECS,
    )

    train_ds = get_dataset("train")
    val_ds = get_dataset("valid")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    num_train = len(train_ds)
    num_val = len(val_ds)
    steps_per_epoch = max(1, len(train_loader))
    print("\n=== Diffusion-JEPA Training ===")
    tb_url = _proxy_url(43802, "tb-sage", absolute=True)
    print(f"TensorBoard: {tb_url}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(
        f"Dataset: TinyImageNet train={num_train} val={num_val} image_size={IMAGE_SIZE}"
    )
    print(
        f"Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | Steps/epoch: {steps_per_epoch}"
    )
    print(
        f"Diffusion timesteps: {DIFFUSION_TIMESTEPS} | Inference steps: {INFERENCE_TIMESTEPS}"
    )

    student = Student(
        ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)
    ).to(DEVICE)
    teacher = Teacher(
        ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)
    ).to(DEVICE)
    teacher.ema_update(student, m=0.0)

    diffusion = DiffusionSchedule(DIFFUSION_TIMESTEPS, BETA_START, BETA_END).to(DEVICE)
    predictor = UNetDiffusionPredictor(
        HIDDEN_DIM, TIME_EMB_DIM, UNET_DEPTH, UNET_HIDDEN_MULT, UNET_NUM_HEADS
    ).to(DEVICE)
    xattn = CondCrossAttention(HIDDEN_DIM, num_heads=4).to(DEVICE)

    def build_param_groups(modules):
        decay, no_decay = [], []
        for m in modules:
            for n, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                # Apply WD only to sufficiently-matrix-like weights (e.g., Linear.weight)
                if n.endswith("weight") and p.ndim >= 2:
                    decay.append(p)
                else:
                    # biases, LayerNorm/ActNorm scale/bias, embeddings, etc.
                    no_decay.append(p)
        return [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    param_groups = build_param_groups([student, predictor, xattn])
    opt = torch.optim.AdamW(param_groups, lr=LR)

    writer.add_text(
        "hparams",
        str(
            {
                "PATCH_SIZE": PATCH_SIZE,
                "HIDDEN_DIM": HIDDEN_DIM,
                "VIT_LAYERS": VIT_LAYERS,
                "VIT_HEADS": VIT_HEADS,
                "NUM_TARGET_BLOCKS": NUM_TARGET_BLOCKS,
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "BATCH_SIZE": BATCH_SIZE,
                "IMAGE_SIZE": IMAGE_SIZE,
                "EPOCHS": EPOCHS,
                "DEVICE": str(DEVICE),
                "DIFFUSION_TIMESTEPS": DIFFUSION_TIMESTEPS,
                "UNET_DEPTH": UNET_DEPTH,
            }
        ),
    )

    total_samples = EPOCHS * len(train_ds)
    steps_per_epoch = max(1, len(train_loader))
    samples_seen = 0
    last_log_samples = 0
    last_val_samples = 0
    LOG_INTERVAL_SAMPLES = LOG_INTERVAL_MULTIPLIER * BATCH_SIZE
    VAL_INTERVAL_SAMPLES = VAL_INTERVAL_MULTIPLIER * BATCH_SIZE

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_knn = 0.0
    if os.path.isfile(LAST_CKPT):
        try:
            state = torch.load(LAST_CKPT, map_location=DEVICE)
            student.load_state_dict(state["student"], strict=True)
            teacher.load_state_dict(state["teacher"], strict=True)
            predictor.load_state_dict(state["predictor"], strict=True)
            try:
                xattn.load_state_dict(state["xattn"], strict=True)
            except Exception:
                print(
                    "[resume] Could not load 'xattn' state_dict from checkpoint (may be absent)"
                )
            opt.load_state_dict(state["opt"])
            samples_seen = int(state.get("samples_seen", 0))
            last_log_samples = int(state.get("last_log_samples", samples_seen))
            last_val_samples = int(state.get("last_val_samples", samples_seen))
            best_knn = float(state.get("best_knn", 0.0))
            if "rng" in state:
                try:
                    random.setstate(state["rng"]["py"])
                    np.random.set_state(state["rng"]["np"])
                    torch.set_rng_state(state["rng"]["torch"])
                    if torch.cuda.is_available() and "cuda" in state["rng"]:
                        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
                except Exception:
                    pass
            print(f"Resumed from {LAST_CKPT} at samples_seen={samples_seen}")
        except Exception as e:
            print(f"Could not resume from {LAST_CKPT}: {e}")

    interpreter = ModelInterpreter(image_size=IMAGE_SIZE, device=DEVICE)
    projector2d = FixedProjector2D(perc=PROJECTOR_PERCENTILE)
    val_feats_prev0 = None
    tracked_idx = None
    prev_grad_vec = None

    steps_done = 0
    for epoch in range(EPOCHS):
        student.train()
        predictor.train()
        xattn.train()
        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            x = x.to(DEVICE, non_blocking=True)
            B = x.size(0)
            grid = student.backbone.grid
            T = grid * grid

            ctx_list, tgt_list = [], []
            for _b in range(B):
                ctx_mask, tgt_mask = sample_context_and_targets(grid)
                ctx_list.append(ctx_mask)
                tgt_list.append(tgt_mask)
            context_mask = torch.stack(ctx_list, dim=0).to(x.device)
            target_mask = torch.stack(tgt_list, dim=0).to(x.device)

            with torch.no_grad():
                t_tokens = teacher(x)

            tgt_idxs = [
                target_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(B)
            ]
            St = int(max([int(idx.numel()) for idx in tgt_idxs]) if tgt_idxs else 0)
            D = t_tokens.size(2)
            teacher_tgt = torch.zeros((B, St, D), device=x.device, dtype=t_tokens.dtype)
            tgt_pad = torch.ones((B, St), dtype=torch.bool, device=x.device)
            pos = student.backbone.pos_embed
            pos_t = torch.zeros((B, St, D), device=x.device, dtype=t_tokens.dtype)
            for i, idx in enumerate(tgt_idxs):
                L = int(idx.numel())
                if L == 0:
                    continue
                teacher_tgt[i, :L] = t_tokens[i, idx]
                tgt_pad[i, :L] = False
                pos_t[i, :L] = pos[0, idx]

            ctx_tokens, ctx_pad = student.encode_context(x, context_mask)
            cond_attn = xattn(ctx_tokens, ctx_pad, pos_t)
            cond = cond_attn + pos_t

            visible = ~tgt_pad
            if visible.any():
                y_flat = teacher_tgt[visible]
                c_flat = cond[visible]

                t_diff = torch.randint(
                    0, diffusion.timesteps, (y_flat.size(0),), device=x.device
                )
                noise = torch.randn_like(y_flat)
                y_noised = diffusion.q_sample(y_flat, t_diff, noise)

                noise_pred = predictor(y_noised, t_diff, c_flat)

                loss = F.mse_loss(noise_pred, noise)
                loss_val = float(loss.item())

                with torch.no_grad():
                    pred_tgt = torch.zeros_like(teacher_tgt)
                    pred_flat = ddim_sample(
                        diffusion, predictor, c_flat, INFERENCE_TIMESTEPS
                    )
                    pred_tgt[visible] = pred_flat

                    mse_emb = F.mse_loss(pred_flat, y_flat, reduction="mean").item()
                    cos_sim = (
                        F.cosine_similarity(pred_flat, y_flat, dim=-1).mean().item()
                    )
            else:
                loss = teacher_tgt.new_tensor(0.0)
                loss_val = float(0.0)
                mse_emb = float(0.0)
                cos_sim = float(0.0)
                pred_tgt = torch.zeros_like(teacher_tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                list(student.parameters())
                + list(predictor.parameters())
                + list(xattn.parameters()),
                GRAD_CLIP_MAX_NORM,
            )
            opt.step()
            curr_lr = float(opt.param_groups[0].get("lr", LR))
            curr_wd = float(opt.param_groups[0].get("weight_decay", WEIGHT_DECAY))

            try:
                grads = []
                for p in (
                    list(student.parameters())
                    + list(predictor.parameters())
                    + list(xattn.parameters())
                ):
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                if grads:
                    gvec = torch.cat(grads)
                    pos_ratio = float((gvec > 0).float().mean().item())
                    cos_to_prev = float("nan")
                    if (
                        prev_grad_vec is not None
                        and prev_grad_vec.numel() == gvec.numel()
                    ):
                        a = F.normalize(gvec.float(), dim=0)
                        b = F.normalize(prev_grad_vec.float(), dim=0)
                        cos_to_prev = float((a * b).sum().item())
                    prev_grad_vec = gvec.detach().clone()
                else:
                    pos_ratio = float("nan")
                    cos_to_prev = float("nan")
            except Exception:
                pos_ratio = float("nan")
                cos_to_prev = float("nan")

            m = momentum_schedule(samples_seen, total_samples)
            teacher.ema_update(student, m)

            samples_seen += B
            steps_done += 1

            if (samples_seen - last_log_samples) >= LOG_INTERVAL_SAMPLES:
                last_log_samples = samples_seen
                writer.add_scalar("train/loss", loss_val, global_step=samples_seen)
                try:
                    writer.add_scalar(
                        "train/grad_norm", float(gn), global_step=samples_seen
                    )
                    writer.add_scalar("train/lr", curr_lr, global_step=samples_seen)
                    writer.add_scalar(
                        "train/weight_decay", curr_wd, global_step=samples_seen
                    )
                    writer.add_scalar(
                        "train/grad_pos_ratio", pos_ratio, global_step=samples_seen
                    )
                    writer.add_scalar(
                        "train/grad_cos_to_prev", cos_to_prev, global_step=samples_seen
                    )
                    writer.add_scalar(
                        "train/mse_embedding", mse_emb, global_step=samples_seen
                    )
                    writer.add_scalar(
                        "train/cos_sim", cos_sim, global_step=samples_seen
                    )
                    writer.add_scalars(
                        "loss", {"train": loss_val}, global_step=samples_seen
                    )
                except Exception:
                    pass
                try:
                    scal = compute_diag_metrics(
                        pred_tgt, teacher_tgt, tgt_pad, context_mask, target_mask
                    )
                    tb_log_scalars(writer, samples_seen, scal)
                    if (samples_seen // LOG_INTERVAL_SAMPLES) % 10 == 0:
                        tb_log_mask_overlay(
                            writer,
                            "viz/masks",
                            samples_seen,
                            x[:1],
                            context_mask,
                            target_mask,
                            grid,
                        )
                        x_vis = x[:1].detach().cpu()
                        fig = interpreter.vit_attention(
                            student.backbone,
                            nn.Sequential(student.backbone),
                            x_vis,
                            layer=-1,
                            alpha=0.5,
                            title="ViT Attention",
                        )
                        import plotly.io as pio, imageio, io, numpy as _np

                        png = pio.to_image(fig, format="png", width=600, height=600)
                        img = imageio.v2.imread(io.BytesIO(png))
                        writer.add_image(
                            "viz/vit_attention",
                            _np.transpose(img, (2, 0, 1)),
                            global_step=samples_seen,
                        )
                        tb_log_vector_projection(
                            writer,
                            "vectors/pca",
                            samples_seen,
                            pred_tgt,
                            teacher_tgt,
                            tgt_pad,
                        )
                    if (
                        samples_seen // LOG_INTERVAL_SAMPLES
                    ) % 20 == 0 and "gvec" in locals():
                        try:
                            writer.add_histogram(
                                "grads/global",
                                gvec.detach().cpu().numpy(),
                                global_step=samples_seen,
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                print(
                    f"[train] epoch {epoch+1}/{EPOCHS} step {batch_idx}/{steps_per_epoch}: loss={loss.item():.4f}, mse_emb={mse_emb:.4f}"
                )

            if (samples_seen - last_val_samples) >= VAL_INTERVAL_SAMPLES:
                last_val_samples = samples_seen
                teacher.backbone.eval()
                print("[val] building feature banks for kNN …")
                bank_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=BANK_NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                )
                tr_feats, tr_labels = extract_global_embeddings(
                    teacher.backbone, bank_loader, DEVICE
                )
                va_feats, va_labels = extract_global_embeddings(
                    teacher.backbone, val_loader, DEVICE
                )
                knn_top1 = knn_eval(tr_feats, tr_labels, va_feats, va_labels, k=KNN_K)
                writer.add_scalar(
                    "val/knn_top1", float(knn_top1), global_step=samples_seen
                )
                print(
                    f"[val] epoch {epoch+1}/{EPOCHS}: loss={loss.item():.4f}, kNN@20={knn_top1:.4f}"
                )

                try:
                    val_loss_acc = 0.0
                    val_mse_acc = 0.0
                    val_cos_acc = 0.0
                    n_batches = 0
                    with torch.no_grad():
                        for vb_idx, (vx, _) in enumerate(val_loader):
                            if vb_idx >= VAL_LOSS_MAX_BATCHES:
                                break
                            vx = vx.to(DEVICE)
                            Bv = vx.size(0)
                            grid_v = student.backbone.grid
                            ctx_list_v, tgt_list_v = [], []
                            for _b in range(Bv):
                                cm, tm = sample_context_and_targets(grid_v)
                                ctx_list_v.append(cm)
                                tgt_list_v.append(tm)
                            cmask = torch.stack(ctx_list_v, dim=0).to(vx.device)
                            tmask = torch.stack(tgt_list_v, dim=0).to(vx.device)
                            ttok = teacher(vx)
                            tidx = [
                                tmask[i].nonzero(as_tuple=False).squeeze(1)
                                for i in range(Bv)
                            ]
                            Stv = int(
                                max([int(ix.numel()) for ix in tidx]) if tidx else 0
                            )
                            Dv = ttok.size(2)
                            teach_v = torch.zeros(
                                (Bv, Stv, Dv), device=vx.device, dtype=ttok.dtype
                            )
                            tpad = torch.ones(
                                (Bv, Stv), dtype=torch.bool, device=vx.device
                            )
                            pos = student.backbone.pos_embed
                            pos_tv = torch.zeros(
                                (Bv, Stv, Dv), device=vx.device, dtype=ttok.dtype
                            )
                            for i, ix in enumerate(tidx):
                                Lv = int(ix.numel())
                                if Lv == 0:
                                    continue
                                teach_v[i, :Lv] = ttok[i, ix]
                                tpad[i, :Lv] = False
                                pos_tv[i, :Lv] = pos[0, ix]
                            ctx_tok, ctx_pad = student.encode_context(vx, cmask)
                            cond_attn_v = xattn(ctx_tok, ctx_pad, pos_tv)
                            cond_v = cond_attn_v + pos_tv
                            visible_v = ~tpad
                            if visible_v.any():
                                yv_flat = teach_v[visible_v]
                                cv_flat = cond_v[visible_v]
                                t_diff_v = torch.randint(
                                    0,
                                    diffusion.timesteps,
                                    (yv_flat.size(0),),
                                    device=vx.device,
                                )
                                noise_v = torch.randn_like(yv_flat)
                                yv_noised = diffusion.q_sample(
                                    yv_flat, t_diff_v, noise_v
                                )
                                noise_pred_v = predictor(yv_noised, t_diff_v, cv_flat)
                                vloss = F.mse_loss(noise_pred_v, noise_v).item()
                                val_loss_acc += vloss
                                pred_v_flat = ddim_sample(
                                    diffusion, predictor, cv_flat, INFERENCE_TIMESTEPS
                                )
                                val_mse_acc += F.mse_loss(pred_v_flat, yv_flat).item()
                                val_cos_acc += (
                                    F.cosine_similarity(pred_v_flat, yv_flat, dim=-1)
                                    .mean()
                                    .item()
                                )
                                n_batches += 1
                    if n_batches > 0:
                        vloss = val_loss_acc / n_batches
                        writer.add_scalar("val/loss", vloss, global_step=samples_seen)
                        writer.add_scalar(
                            "val/mse_embedding",
                            val_mse_acc / n_batches,
                            global_step=samples_seen,
                        )
                        writer.add_scalar(
                            "val/cos_sim",
                            val_cos_acc / n_batches,
                            global_step=samples_seen,
                        )
                        writer.add_scalars(
                            "loss", {"val": float(vloss)}, global_step=samples_seen
                        )
                except Exception:
                    pass

                try:
                    val_feats_prev0, tracked_idx = tb_log_knn_convergence(
                        writer,
                        "viz/knn_convergence",
                        samples_seen,
                        projector2d,
                        tr_feats,
                        val_feats_prev0,
                        va_feats,
                        tracked_idx,
                    )
                except Exception:
                    pass

                ckpt_state: Dict[str, Any] = {
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "predictor": predictor.state_dict(),
                    "xattn": xattn.state_dict(),
                    "opt": opt.state_dict(),
                    "samples_seen": samples_seen,
                    "steps_done": steps_done,
                    "last_log_samples": last_log_samples,
                    "last_val_samples": last_val_samples,
                    "best_knn": best_knn,
                    "hparams": {
                        "PATCH_SIZE": PATCH_SIZE,
                        "HIDDEN_DIM": HIDDEN_DIM,
                        "VIT_LAYERS": VIT_LAYERS,
                        "VIT_HEADS": VIT_HEADS,
                        "NUM_TARGET_BLOCKS": NUM_TARGET_BLOCKS,
                        "LR": LR,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "DIFFUSION_TIMESTEPS": DIFFUSION_TIMESTEPS,
                    },
                    "rng": {
                        "py": random.getstate(),
                        "np": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                    },
                }
                torch.save(ckpt_state, LAST_CKPT)
                print(f"[ckpt] saved last checkpoint → {LAST_CKPT}")
                if knn_top1 > best_knn:
                    best_knn = knn_top1
                    ckpt_state["best_knn"] = best_knn
                    torch.save(ckpt_state, BEST_CKPT)
                    print(f"[ckpt] new best kNN {best_knn:.4f} → {BEST_CKPT}")

    final_state = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "predictor": predictor.state_dict(),
        "xattn": xattn.state_dict(),
        "opt": opt.state_dict(),
        "samples_seen": samples_seen,
        "best_knn": best_knn,
    }
    torch.save(final_state, LAST_CKPT)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    train_run()
