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
    tb_log_flow_samples_2d,
    tb_log_flow_stats,
    tb_log_flow_hist,
)

# ------------------
# Constants / Defaults
# ------------------
NUM_CLASSES = 100
IMAGE_SIZE = 64
EPOCHS = 400
BATCH_SIZE = 600
LR = 5e-5                     # Small constant learning rate
WEIGHT_DECAY = 1e-4
DEVICE_TYPE = (
    'mps'
    if torch.backends.mps.is_available()
    else ('cuda' if torch.cuda.is_available() else 'cpu')
)
DEVICE = torch.device(DEVICE_TYPE)

# Logging / runtime config
RUN_ID = time.strftime('%Y%m%d-%H%M%S')

# DataLoader parameters
TRAIN_NUM_WORKERS = 2
VAL_NUM_WORKERS = 2
BANK_NUM_WORKERS = 2
PIN_MEMORY = (DEVICE_TYPE == 'cuda')

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


# JEPA loss weights
LOSS_ALIGN_WEIGHT = 1.0
LOSS_VAR_WEIGHT = 0.1
LOSS_COV_WEIGHT = 0.01
LOSS_STD_FLOOR = 0.5

# ------------------
# Normalizing Flow (RealNVP-style) config
# ------------------
# RealNVP is the default head here (no feature flag).
NF_LAYERS = 6                     # Number of affine coupling steps
NF_HIDDEN = 768                   # Hidden size in s,t MLPs
NF_S_CLAMP = 2.0                  # Bound on log-scale via tanh to avoid overflow
COND_ACTNORM_EPS = 1e-6           # Numerical floor for ActNorm over cond vectors

# Masking / targets
NUM_TARGET_BLOCKS = 4
TARGET_SCALE = (0.15, 0.20)   # relative area wrt image
TARGET_AR = (0.75, 1.50)      # aspect ratio range
MIN_CTX_PATCHES = 8           # ensure context has at least this many tokens

# EMA
MOMENTUM_INIT = 0.998
MOMENTUM_FINAL = 1.0

CHECKPOINT_DIR = os.path.abspath('./checkpoints')
LAST_CKPT = os.path.join(CHECKPOINT_DIR, '2_ijeja_last.pt')
BEST_CKPT = os.path.join(CHECKPOINT_DIR, '2_ijeja_best_knn.pt')

TB_LOGDIR = os.path.abspath('./runs')
TB_PORT = 43802
TB_SUFFIX = 'tb-sage'
TB_FLUSH_SECS = 5

# Evaluation / diagnostics
KNN_K = 20
VAL_LOSS_MAX_BATCHES = 5
PROJECTOR_PERCENTILE = 99.0

# No LR scheduler -> keep constants minimal

def _proxy_url(port: int, suffix: str, absolute: bool = True) -> str:
    rel = f"/proxy/absolute/{port}/{suffix}/"
    if not absolute:
        return rel
    root = (
        os.environ.get('JUPYTERHUB_ROOT_URL') or
        os.environ.get('NB_URL') or
        os.environ.get('NB_HOST') or
        os.environ.get('AIM_UI_ORIGIN') or 'https://45957f89c897.innodatahub.innopolis.university'
    )
    root = root.rstrip('/')
    return f"{root}{rel}"


def start_tb_ui(logdir: str = os.path.abspath('./runs'), port: int = 43802, suffix: str = 'tb-sage') -> Optional[str]:
    try:
        from IPython.display import HTML, display
        IN_NOTEBOOK = ('ipykernel' in sys.modules)
    except Exception:
        HTML = None; display = None; IN_NOTEBOOK = False

    os.makedirs(logdir, exist_ok=True)
    try:
        subprocess.run(['bash', '-lc', f"pkill -f 'tensorboard .*{port}' || true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    prefix = f"/proxy/absolute/{port}/{suffix}"
    try:
        proc = subprocess.Popen(['tensorboard','--logdir',logdir,'--host','127.0.0.1','--port',str(port),'--path_prefix',prefix], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print("tensorboard не найден. Установите пакет 'tensorboard'.")
        return None

    def port_open(p):
        s = socket.socket(); s.settimeout(0.2)
        try:
            s.connect(('127.0.0.1', p)); s.close(); return True
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
            display(HTML(f'<a href="{abs_url}" target="_blank">открыть TensorBoard</a>'))
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
        return img.convert('RGB')


class TinyImageNetTorch(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]['image']
        label = self.data[index]['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def _filter_label_lt_num_classes(example):
    return int(example.get('label', -1)) < NUM_CLASSES


def get_dataset(split: str) -> TinyImageNetTorch:
    def load_tinyimagenet(_split):
        dataset = load_dataset('zh-plus/tiny-imagenet', split=_split)
        dataset = dataset.filter(_filter_label_lt_num_classes)
        return dataset

    def compute_statistics(dataset):
        n_channels = 3
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        n_samples = len(dataset)
        for sample in dataset:
            img = sample['image']
            img = transforms.ToTensor()(img)
            mean += img.mean(dim=(1, 2))
            std += img.std(dim=(1, 2))
        mean /= n_samples
        std /= n_samples
        return mean.numpy(), std.numpy()

    global TRAIN_MEAN_STD
    if TRAIN_MEAN_STD is None:
        train_raw = load_tinyimagenet('train')
        TRAIN_MEAN_STD = compute_statistics(train_raw)
    mean, std = TRAIN_MEAN_STD

    raw_dataset = load_tinyimagenet(split)

    if split == 'train':
        tfms = transforms.Compose([
            ToRGB(),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])
    else:
        tfms = transforms.Compose([
            ToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])
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
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # key_padding_mask: [B, S], True where to ignore
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(self, image_size: int, patch_size: int, hidden_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        assert image_size % patch_size == 0
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        self.transformer_layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> tokens [B, T, D]
        t = self.patch_embed(x)                 # [B, D, g, g]
        t = t.flatten(2).transpose(1, 2)        # [B, T, D]
        t = t + self.pos_embed
        for blk in self.transformer_layers:
            t = blk(t, key_padding_mask=None)
        return self.norm(t)                     # [B, T, D]


# -----------------
# Student / Teacher
# -----------------
class Predictor(nn.Module):
    """Narrow ViT predictor over [context tokens | target queries].
    Inputs are already position-encoded. Returns outputs for target slice only.
    """
    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.q_token = nn.Parameter(torch.zeros(dim))  # learnable target query token
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, ctx: torch.Tensor, ctx_pad: torch.Tensor, tgt_q: torch.Tensor, tgt_pad: torch.Tensor) -> torch.Tensor:
        # ctx: [B, Sc, D], ctx_pad: [B, Sc] True=pad
        # tgt_q: [B, St, D], tgt_pad: [B, St] True=pad
        x = torch.cat([ctx, tgt_q], dim=1)  # [B, Sc+St, D]
        pad = torch.cat([ctx_pad, tgt_pad], dim=1)  # [B, Sc+St]
        for blk in self.layers:
            x = blk(x, key_padding_mask=pad)
        x = self.norm(x)
        Sc = ctx.size(1)
        return x[:, Sc:, :]  # [B, St, D] predictions for target tokens


class Student(nn.Module):
    def __init__(self, backbone: ViTBackbone):
        super().__init__()
        self.backbone = backbone

    def encode_context(self, x: torch.Tensor, context_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # context_mask: [B, T] True where context is visible; return padded ctx tokens + pad mask
        with torch.no_grad():
            B = x.size(0)
        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)  # [B,T,D]
        # gather context indices per-sample and pad
        pos = self.backbone.pos_embed  # [1,T,D]
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
        # run transformer layers with key padding mask
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
        return self.backbone(x)  # [B,T,D]

    @torch.no_grad()
    def ema_update(self, student: Student, m: float):
        for p_t, p_s in zip(self.backbone.parameters(), student.backbone.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))

# --------------------------------------------
# Conditional Normalizing Flow (RealNVP-style)
# --------------------------------------------
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

    def forward(self, ctx_tokens: torch.Tensor, ctx_pad: torch.Tensor, tgt_pos: torch.Tensor) -> torch.Tensor:
        # ctx_tokens: [B, Sc, D], ctx_pad: [B, Sc] (True=pad), tgt_pos: [B, St, D]
        B, St, D = tgt_pos.size()
        # broadcast learned target token to [B, St, D]
        tgt_tok = self.tgt_token.expand(B, St, D)
        query = tgt_pos + tgt_tok                      # [B, St, D]
        # MultiheadAttention with batch_first=True
        # key_padding_mask: [B, Sc], True = ignore
        h, _ = self.attn(query, ctx_tokens, ctx_tokens, key_padding_mask=ctx_pad)  # [B, St, D]
        h = self.ln(h)
        return h

class AffineCoupling(nn.Module):
    """
    One RealNVP affine coupling layer with conditional s(x_A, c), t(x_A, c).
    Split dims into A (masked=1) and B (masked=0). Keep A, transform B:
      y_A = x_A
      y_B = x_B * exp(s) + t
    log|det J| = sum(s) over transformed dims.
    """
    def __init__(self, dim: int, cond_dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer('mask', mask.view(1, dim))  # [1,D] with {0,1}
        self.dim = dim
        self.cond_dim = cond_dim
        self.hidden = hidden
        dA = int(mask.sum().item())
        dB = dim - dA
        self.net = nn.Sequential(
            nn.Linear(dA + cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * dB)
        )
        # Zero-init the last layer to start near identity (stable w/o tanh clamp)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [N,D], cond: [N,cond_dim]
        m = self.mask  # [1,D]
        xA = x[:, m[0].bool()]     # [N,dA]
        xB = x[:, (~m[0].bool())]  # [N,dB]
        h = torch.cat([xA, cond], dim=-1)
        st = self.net(h)
        dB = xB.size(1)
        s, t = st[:, :dB], st[:, dB:]
        # Bound log-scale to keep exp(s) in a safe range
        s = NF_S_CLAMP * torch.tanh(s)
        yB = xB * torch.exp(s) + t
        # reconstruct y
        y = x.clone()
        y[:, (~m[0].bool())] = yB
        logdet = s.sum(dim=1)  # [N]
        return y, logdet
    
    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.mask
        yA = y[:, m[0].bool()]
        yB = y[:, (~m[0].bool())]
        h = torch.cat([yA, cond], dim=-1)
        st = self.net(h)
        dB = yB.size(1)
        s, t = st[:, :dB], st[:, dB:]
        s = NF_S_CLAMP * torch.tanh(s)
        xB = (yB - t) * torch.exp(-s)
        x = y.clone()
        x[:, (~m[0].bool())] = xB
        logdet = (-s).sum(dim=1)  # inverse log-det
        return x, logdet


class ActNorm1D(nn.Module):
    """Data-dependent affine normalization over final dimension (Glow-style)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))

    def _initialize(self, x: torch.Tensor):
        flat = x.reshape(-1, self.dim)
        if flat.numel() == 0:
            return
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        with torch.no_grad():
            self.bias.copy_(-mean)
            self.log_scale.copy_(torch.log(1.0 / std))
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.item()) and self.training:
            self._initialize(x.detach())
        scale = torch.exp(self.log_scale)
        return (x + self.bias) * scale


# ---- New: FlowActNorm1D for flows with exact log-det ----
class FlowActNorm1D(nn.Module):
    """
    Glow-style ActNorm for flows over feature dimension with exact log-det.
    Acts on shape [N, D]. Uses data-dependent init on first training batch.
    y = (x + bias) * exp(log_scale)
    log|det J| per-sample = sum(log_scale)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))

    def _initialize(self, x: torch.Tensor):
        flat = x.reshape(-1, self.dim)
        if flat.numel() == 0:
            return
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        with torch.no_grad():
            self.bias.copy_(-mean)
            self.log_scale.copy_(torch.log(1.0 / std))
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.initialized.item()) and self.training:
            self._initialize(x.detach())
        y = (x + self.bias) * torch.exp(self.log_scale)
        # log-det per sample: sum over features of log_scale
        logdet = torch.sum(self.log_scale, dim=0).expand(x.size(0)).to(x.dtype).to(x.device)
        return y, logdet

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = y * torch.exp(-self.log_scale) - self.bias
        logdet = torch.sum(-self.log_scale, dim=0).expand(y.size(0)).to(y.dtype).to(y.device)
        return x, logdet


class FlowStep(nn.Module):
    """One coupling + actnorm step with forward/inverse and log-det tracking."""
    def __init__(self, dim: int, cond_dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.coupling = AffineCoupling(dim, cond_dim, hidden, mask)
        self.an = FlowActNorm1D(dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, ld1 = self.coupling(z, cond)
        x, ld2 = self.an(x)
        return x, ld1 + ld2

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, ld2 = self.an.inverse(x)
        z, ld1 = self.coupling.inverse(y, cond)
        return z, ld1 + ld2

class CondRealNVPFlow(nn.Module):
    """
    Conditional RealNVP over D-dimensional target tokens with context conditioning.
    - Base z ~ N(0,I_D)
    - Flow x = Phi_theta^1(z | cond), cond = pool(ctx) + pos(target_idx)
    Training via exact log-likelihood:
      log p_theta(x|cond) = log p0(z) + sum_k log|det J_k|,  where z = Phi^{-1}(x|cond)
    """
    def __init__(self, dim: int, cond_dim: int, num_layers: int = NF_LAYERS, hidden: int = NF_HIDDEN):
        super().__init__()
        masks = []
        # Random binary masks per step (~half dims active)
        for _ in range(num_layers):
            mask = torch.zeros(dim)
            idx = torch.randperm(dim)[: dim // 2]
            mask[idx] = 1.0
            masks.append(mask)
        self.layers = nn.ModuleList([FlowStep(dim, cond_dim, hidden, m) for m in masks])
        self.dim = dim

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z -> x, accumulate log-det
        logdet = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        x = z
        for layer in self.layers:
            x, ld = layer(x, cond)
            logdet = logdet + ld
        return x, logdet
    
    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x -> z, accumulate log-det of inverse
        logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        z = x
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, cond)
            logdet = logdet + ld
        return z, logdet
    
    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Exact log-likelihood under the flow
        z, logdet = self.inverse(x, cond) # z ~ approx N(0,I)
        # log p0(z) for standard normal
        log_p0 = -0.5 * (z.pow(2).sum(dim=1) + self.dim * math.log(2.0 * math.pi))
        return log_p0 + logdet # [N]
    
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        # Draw one sample per cond row
        N = cond.size(0)
        z = torch.randn(N, self.dim, device=cond.device, dtype=cond.dtype)
        x, _ = self.forward(z, cond)
        return x


# --------------
# Mask utilities
# --------------
def _sample_blocks(grid: int, n_blocks: int, scale: Tuple[float, float], ar: Tuple[float, float]) -> List[Tuple[int, int, int, int]]:
    # Return list of blocks as (y0, x0, h, w) in patch units
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


def build_mask_from_blocks(grid: int, blocks: List[Tuple[int, int, int, int]]) -> torch.Tensor:
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    for (y0, x0, h, w) in blocks:
        mask[y0:y0 + h, x0:x0 + w] = True
    return mask.flatten()  # [T]


def sample_context_and_targets(grid: int, min_ctx: int = MIN_CTX_PATCHES, max_tries: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample context window and target blocks; resample until context non-empty."""
    for _ in range(max_tries):
        ctx_block = _sample_blocks(grid, 1, (0.85, 1.0), (0.75, 1.5))[0]
        tgt_blocks = _sample_blocks(grid, NUM_TARGET_BLOCKS, TARGET_SCALE, TARGET_AR)
        ctx = torch.zeros(grid, grid, dtype=torch.bool)
        y0, x0, h, w = ctx_block
        ctx[y0:y0 + h, x0:x0 + w] = True
        tgt = torch.zeros(grid, grid, dtype=torch.bool)
        for (yy, xx, hh, ww) in tgt_blocks:
            tgt[yy:yy + hh, xx:xx + ww] = True
        ctx[tgt] = False
        if int(ctx.sum().item()) >= min_ctx and int(tgt.sum().item()) > 0:
            return ctx.flatten(), tgt.flatten()
    # Fallback: whole image context minus smaller targets
    ctx = torch.ones(grid, grid, dtype=torch.bool)
    tgt_blocks = _sample_blocks(grid, max(1, NUM_TARGET_BLOCKS // 2), (0.10, 0.15), TARGET_AR)
    tgt = torch.zeros(grid, grid, dtype=torch.bool)
    for (yy, xx, hh, ww) in tgt_blocks:
        tgt[yy:yy + hh, xx:xx + ww] = True
    ctx[tgt] = False
    if int(ctx.sum().item()) == 0:
        ctx[0, 0] = True; tgt[0, 0] = False
    return ctx.flatten(), tgt.flatten()


# ----------------
# Loss & training
# ----------------
def _visible_tokens(pred_tgt: torch.Tensor, tgt_pad: torch.Tensor) -> Optional[torch.Tensor]:
    """Return normalized visible tokens flattened across batch."""
    visible = ~tgt_pad
    if not torch.any(visible):
        return None
    z = F.normalize(pred_tgt, dim=-1)
    z = z[visible]
    if z.numel() == 0:
        return None
    return z


def metric_alignment(pred_tgt: torch.Tensor, teacher_tgt: torch.Tensor, tgt_pad: torch.Tensor) -> torch.Tensor:
    """Angular alignment between student predictions and teacher targets."""
    mask = (~tgt_pad).float()
    denom = mask.sum()
    if denom.item() == 0:
        return pred_tgt.new_tensor(0.0)
    p = F.normalize(pred_tgt, dim=-1)
    t = F.normalize(teacher_tgt, dim=-1)
    diff = (p - t).pow(2).sum(dim=-1)
    return (diff * mask).sum() / denom


def metric_variance(pred_tgt: torch.Tensor, tgt_pad: torch.Tensor, std_floor: float = LOSS_STD_FLOOR) -> torch.Tensor:
    """Variance term keeps per-dimension std above a floor."""
    z = _visible_tokens(pred_tgt, tgt_pad)
    if z is None:
        return pred_tgt.new_tensor(0.0)
    std = z.std(dim=0, unbiased=False) + 1e-4
    return F.relu(std_floor - std).pow(2).mean()


def metric_covariance(pred_tgt: torch.Tensor, tgt_pad: torch.Tensor) -> torch.Tensor:
    """Decorrelation term penalizes off-diagonal covariance."""
    z = _visible_tokens(pred_tgt, tgt_pad)
    if z is None or z.size(0) <= 1:
        return pred_tgt.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.t() @ z) / max(1, z.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    if off_diag.numel() == 0:
        return pred_tgt.new_tensor(0.0)
    return off_diag.pow(2).mean()


def jepa_metrics(
    pred_tgt: torch.Tensor,
    teacher_tgt: torch.Tensor,
    tgt_pad: torch.Tensor,
    std_floor: float = LOSS_STD_FLOOR,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    align = metric_alignment(pred_tgt, teacher_tgt, tgt_pad)
    var = metric_variance(pred_tgt, tgt_pad, std_floor)
    cov = metric_covariance(pred_tgt, tgt_pad)
    return align, var, cov


def momentum_schedule(it: int, total_its: int, m0: float = MOMENTUM_INIT, m1: float = MOMENTUM_FINAL) -> float:
    # cosine schedule to 1.0
    if total_its <= 1:
        return m1
    t = min(1.0, max(0.0, it / (total_its - 1)))
    return m1 - (m1 - m0) * (0.5 * (1.0 + math.cos(math.pi * t)))

# (WD stays constant under AdamW; adjust WEIGHT_DECAY if needed)

@torch.no_grad()
def extract_global_embeddings(backbone: ViTBackbone, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Return features [N,D] via mean pooling tokens, and labels [N]
    backbone.eval()
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        t = backbone(x)          # [B,T,D]
        g = t.mean(dim=1)        # [B,D]
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
    # cosine similarity kNN
    train_norm = F.normalize(train_feats, dim=1)
    val_norm = F.normalize(val_feats, dim=1)
    sim = val_norm @ train_norm.t()  # [Nv, Nt]
    topk = sim.topk(k, dim=1).indices  # [Nv, k]
    neighbors = train_labels[topk]     # [Nv, k]
    # majority vote
    preds = torch.mode(neighbors, dim=1).values
    acc = (preds == val_labels).float().mean().item()
    return float(acc)

def train_run():
    tb_url = start_tb_ui(logdir=os.path.abspath('./runs'), port=43802, suffix='tb-sage')
    writer = SummaryWriter(log_dir=os.path.join(os.path.abspath('./runs'), RUN_ID), flush_secs=TB_FLUSH_SECS)

    # Data
    train_ds = get_dataset('train')
    val_ds = get_dataset('valid')
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
    print("\n=== i-JEPA Training ===")
    tb_url = _proxy_url(43802, 'tb-sage', absolute=True)
    print(f"TensorBoard: {tb_url}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Dataset: TinyImageNet train={num_train} val={num_val} image_size={IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | Steps/epoch: {steps_per_epoch}")

    # Models
    student = Student(ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)).to(DEVICE)
    teacher = Teacher(ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)).to(DEVICE)
    # init teacher = student
    teacher.ema_update(student, m=0.0)
    nf = CondRealNVPFlow(dim=HIDDEN_DIM, cond_dim=HIDDEN_DIM, num_layers=NF_LAYERS, hidden=NF_HIDDEN).to(DEVICE)
    xattn = CondCrossAttention(HIDDEN_DIM, num_heads=4).to(DEVICE)
    cond_norm = ActNorm1D(HIDDEN_DIM, eps=COND_ACTNORM_EPS).to(DEVICE)

    def build_param_groups(modules):
        decay, no_decay = [], []
        for m in modules:
            for n, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                # Apply WD only to sufficiently-matrix-like weights (e.g., Linear.weight)
                if n.endswith('weight') and p.ndim >= 2:
                    decay.append(p)
                else:
                    # biases, LayerNorm/ActNorm scale/bias, embeddings, etc.
                    no_decay.append(p)
        return [
            {'params': decay, 'weight_decay': WEIGHT_DECAY},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

    param_groups = build_param_groups([student, nf, xattn, cond_norm])
    opt = torch.optim.AdamW(param_groups, lr=LR)

    writer.add_text('hparams', str({
        'PATCH_SIZE': PATCH_SIZE,
        'HIDDEN_DIM': HIDDEN_DIM,
        'VIT_LAYERS': VIT_LAYERS,
        'VIT_HEADS': VIT_HEADS,
        'NUM_TARGET_BLOCKS': NUM_TARGET_BLOCKS,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'BATCH_SIZE': BATCH_SIZE,
        'IMAGE_SIZE': IMAGE_SIZE,
        'EPOCHS': EPOCHS,
        'DEVICE': str(DEVICE),
    }))

    # Sample/step accounting
    total_samples = EPOCHS * len(train_ds)
    steps_per_epoch = max(1, len(train_loader))
    samples_seen = 0
    last_log_samples = 0
    last_val_samples = 0
    LOG_INTERVAL_SAMPLES = LOG_INTERVAL_MULTIPLIER * BATCH_SIZE
    VAL_INTERVAL_SAMPLES = VAL_INTERVAL_MULTIPLIER * BATCH_SIZE

    # Resume from checkpoint if present
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.isfile(LAST_CKPT):
        try:
            state = torch.load(LAST_CKPT, map_location=DEVICE)
            student.load_state_dict(state['student'], strict=True)
            teacher.load_state_dict(state['teacher'], strict=True)
            try:
                nf.load_state_dict(state['nf'], strict=True)
            except Exception:
                print("[resume] Could not load 'nf' state_dict from checkpoint (may be absent)")
            try:
                xattn.load_state_dict(state['xattn'], strict=True)
            except Exception:
                print("[resume] Could not load 'xattn' state_dict from checkpoint (may be absent)")
            try:
                cond_norm.load_state_dict(state['cond_norm'], strict=True)
            except Exception:
                print("[resume] Could not load 'cond_norm' state_dict from checkpoint (may be absent)")
            opt.load_state_dict(state['opt'])
            samples_seen = int(state.get('samples_seen', 0))
            steps_done = int(state.get('steps_done', samples_seen // max(1, BATCH_SIZE)))
            last_log_samples = int(state.get('last_log_samples', samples_seen))
            last_val_samples = int(state.get('last_val_samples', samples_seen))
            best_knn = float(state.get('best_knn', 0.0))
            # RNG states
            if 'rng' in state:
                try:
                    random.setstate(state['rng']['py'])
                    np.random.set_state(state['rng']['np'])
                    torch.set_rng_state(state['rng']['torch'])
                    if torch.cuda.is_available() and 'cuda' in state['rng']:
                        torch.cuda.set_rng_state_all(state['rng']['cuda'])
                except Exception:
                    pass
            print(f"Resumed from {LAST_CKPT} at samples_seen={samples_seen}")
        except Exception as e:
            print(f"Could not resume from {LAST_CKPT}: {e}")
            best_knn = 0.0
    else:
        best_knn = 0.0

    interpreter = ModelInterpreter(image_size=IMAGE_SIZE, device=DEVICE)
    # Fixed 2D projector for kNN convergence visualization
    projector2d = FixedProjector2D(perc=PROJECTOR_PERCENTILE)
    val_feats_prev0 = None
    tracked_idx = None
    prev_grad_vec = None

    steps_done = 0
    for epoch in range(EPOCHS):
        student.train(); nf.train(); xattn.train(); cond_norm.train()
        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            x = x.to(DEVICE, non_blocking=True)
            B = x.size(0)
            grid = student.backbone.grid
            T = grid * grid
            # sample context + targets per sample
            ctx_list, tgt_list = [], []
            for _b in range(B):
                ctx_mask, tgt_mask = sample_context_and_targets(grid)
                ctx_list.append(ctx_mask)
                tgt_list.append(tgt_mask)
            context_mask = torch.stack(ctx_list, dim=0).to(x.device)  # [B,T]
            target_mask = torch.stack(tgt_list, dim=0).to(x.device)   # [B,T]

            # Teacher tokens
            with torch.no_grad():
                t_tokens = teacher(x)            # [B,T,D]
            # Gather target teacher tokens and build per-target positional encodings
            tgt_idxs = [target_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(B)]
            St = int(max([int(idx.numel()) for idx in tgt_idxs]) if tgt_idxs else 0)
            D = t_tokens.size(2)
            teacher_tgt = torch.zeros((B, St, D), device=x.device, dtype=t_tokens.dtype)
            tgt_pad = torch.ones((B, St), dtype=torch.bool, device=x.device)
            pos = student.backbone.pos_embed  # [1,T,D]
            pos_t = torch.zeros((B, St, D), device=x.device, dtype=t_tokens.dtype)
            for i, idx in enumerate(tgt_idxs):
                L = int(idx.numel())
                if L == 0:
                    continue
                teacher_tgt[i, :L] = t_tokens[i, idx]
                tgt_pad[i, :L] = False
                pos_t[i, :L] = pos[0, idx]

            # Student encodes only context tokens (padded)
            ctx_tokens, ctx_pad = student.encode_context(x, context_mask)

            # --- Conditional RealNVP: MLE over teacher targets given context ---
            # 1) Build per-target conditioning vectors via cross-attention
            cond_attn = xattn(ctx_tokens, ctx_pad, pos_t)  # [B, St, D]
            cond = cond_attn + pos_t
            # 2) Flatten visible targets and cond
            visible = (~tgt_pad)
            if visible.any():
                cond_vis = cond[visible]
                cond_vis = cond_norm(cond_vis)
                cond[visible] = cond_vis
                y_flat = teacher_tgt[visible]     # [N_vis, D]
                c_flat = cond_vis                 # [N_vis, D]
                # 3) Exact log-likelihood under the flow:
                #    log p_theta(y|c) = log p0(z) + sum log|det J_k|,  where z = Phi^{-1}(y|c)
                logp = nf.log_prob(y_flat, c_flat)            # [N_vis]
                if not torch.isfinite(logp).all():
                    # Skip this batch if instability is detected; avoid poisoning training with NaNs/Infs
                    print("[warn] non-finite logp detected; skipping step")
                    continue
                nll = -logp.mean()
                loss = nll
                loss_val = float(loss.item())
                # For diagnostics comparable to JEPA plots, sample one x_hat per cond:
                with torch.no_grad():
                    yhat_flat = nf.sample(c_flat)             # [N_vis, D]
                # reconstruct padded [B,St,D] for visualization
                pred_tgt = torch.zeros_like(teacher_tgt)
                pred_tgt[visible] = yhat_flat
                # Log cond norms for sanity
                try:
                    cond_norm_mean = float(c_flat.norm(dim=1).mean().item())
                    writer.add_scalar('train/cond_norm_mean', cond_norm_mean, global_step=samples_seen)
                except Exception:
                    pass
                # JEPA-style diagnostics on samples (metrics only; not used in loss)
                align_m, var_m, cov_m = jepa_metrics(pred_tgt, teacher_tgt, tgt_pad, std_floor=LOSS_STD_FLOOR)
                align_val = float(align_m.item())
                var_val = float(var_m.item())
                cov_val = float(cov_m.item())
            else:
                # Fallback when no targets visible (should be rare)
                loss = teacher_tgt.new_tensor(0.0)
                loss_val = float(0.0)
                align_val = float(0.0); var_val = float(0.0); cov_val = float(0.0)

            # Backprop on student + flow + pooler only
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                list(student.parameters())
                + list(nf.parameters())
                + list(xattn.parameters())
                + list(cond_norm.parameters()),
                GRAD_CLIP_MAX_NORM,
            )
            opt.step()
            curr_lr = float(opt.param_groups[0].get('lr', LR))
            curr_wd = float(opt.param_groups[0].get('weight_decay', WEIGHT_DECAY))

            # Gradient diagnostics (directional stability, sign distribution)
            try:
                grads = []
                for p in list(student.parameters()) + list(nf.parameters()) + list(xattn.parameters()) + list(cond_norm.parameters()):
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                if grads:
                    gvec = torch.cat(grads)
                    pos_ratio = float((gvec > 0).float().mean().item())
                    cos_to_prev = float('nan')
                    if prev_grad_vec is not None and prev_grad_vec.numel() == gvec.numel():
                        a = F.normalize(gvec.float(), dim=0)
                        b = F.normalize(prev_grad_vec.float(), dim=0)
                        cos_to_prev = float((a * b).sum().item())
                    prev_grad_vec = gvec.detach().clone()
                else:
                    pos_ratio = float('nan'); cos_to_prev = float('nan')
            except Exception:
                pos_ratio = float('nan'); cos_to_prev = float('nan')

            # EMA update teacher (by sample-based progress)
            m = momentum_schedule(samples_seen, total_samples)
            teacher.ema_update(student, m)

            # Update sample counters
            samples_seen += B
            steps_done += 1

            # Logging by samples (minimal)
            if (samples_seen - last_log_samples) >= LOG_INTERVAL_SAMPLES:
                last_log_samples = samples_seen
                writer.add_scalar('train/nll', loss_val, global_step=samples_seen)
                try:
                    writer.add_scalar('train/grad_norm', float(gn), global_step=samples_seen)
                    writer.add_scalar('train/lr', curr_lr, global_step=samples_seen)
                    writer.add_scalar('train/weight_decay', curr_wd, global_step=samples_seen)
                    writer.add_scalar('train/grad_pos_ratio', pos_ratio, global_step=samples_seen)
                    writer.add_scalar('train/grad_cos_to_prev', cos_to_prev, global_step=samples_seen)
                    writer.add_scalars('loss', {'train': loss_val}, global_step=samples_seen)
                    writer.add_scalars('metrics/components', {'align': align_val, 'var': var_val, 'cov': cov_val}, global_step=samples_seen)
                except Exception:
                    pass
                try:
                    scal = compute_diag_metrics(pred_tgt, teacher_tgt, tgt_pad, context_mask, target_mask)
                    tb_log_scalars(writer, samples_seen, scal)
                    if (samples_seen // LOG_INTERVAL_SAMPLES) % 10 == 0:
                        tb_log_mask_overlay(writer, 'viz/masks', samples_seen, x[:1], context_mask, target_mask, grid)
                        x_vis = x[:1].detach().cpu()
                        fig = interpreter.vit_attention(student.backbone, nn.Sequential(student.backbone), x_vis, layer=-1, alpha=0.5, title='ViT Attention')
                        import plotly.io as pio, imageio, io, numpy as _np
                        png = pio.to_image(fig, format='png', width=600, height=600)
                        img = imageio.v2.imread(io.BytesIO(png))
                        writer.add_image('viz/vit_attention', _np.transpose(img, (2,0,1)), global_step=samples_seen)
                        tb_log_vector_projection(writer, 'vectors/pca', samples_seen, pred_tgt, teacher_tgt, tgt_pad)
                    # Occasional gradient histogram (heavy)
                    if (samples_seen // LOG_INTERVAL_SAMPLES) % 20 == 0 and 'gvec' in locals():
                        try:
                            writer.add_histogram('grads/global', gvec.detach().cpu().numpy(), global_step=samples_seen)
                        except Exception:
                            pass
                except Exception:
                    pass
                print(f"[train] epoch {epoch+1}/{EPOCHS} step {batch_idx}/{steps_per_epoch}: loss={loss.item():.4f}")
            
            # Validation/checkpoint cadence (by samples)
            if (samples_seen - last_val_samples) >= VAL_INTERVAL_SAMPLES:
                last_val_samples = samples_seen
                # kNN evaluation kept (enabled)
                teacher.backbone.eval()
                print("[val] building feature banks for kNN …")
                bank_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=BANK_NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                )
                tr_feats, tr_labels = extract_global_embeddings(teacher.backbone, bank_loader, DEVICE)
                va_feats, va_labels = extract_global_embeddings(teacher.backbone, val_loader, DEVICE)
                knn_top1 = knn_eval(tr_feats, tr_labels, va_feats, va_labels, k=KNN_K)
                writer.add_scalar('val/knn_top1', float(knn_top1), global_step=samples_seen)
                print(f"[val] epoch {epoch+1}/{EPOCHS}: loss={loss.item():.4f}, kNN@20={knn_top1:.4f}")

                # Lightweight validation loss on a few batches
                try:
                    val_loss_acc = 0.0
                    val_align_acc = 0.0
                    val_var_acc = 0.0
                    val_cov_acc = 0.0
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
                                ctx_list_v.append(cm); tgt_list_v.append(tm)
                            cmask = torch.stack(ctx_list_v, dim=0).to(vx.device)
                            tmask = torch.stack(tgt_list_v, dim=0).to(vx.device)
                            ttok = teacher(vx)
                            tidx = [tmask[i].nonzero(as_tuple=False).squeeze(1) for i in range(Bv)]
                            Stv = int(max([int(ix.numel()) for ix in tidx]) if tidx else 0)
                            Dv = ttok.size(2)
                            teach_v = torch.zeros((Bv, Stv, Dv), device=vx.device, dtype=ttok.dtype)
                            tpad = torch.ones((Bv, Stv), dtype=torch.bool, device=vx.device)
                            pos = student.backbone.pos_embed
                            pos_tv = torch.zeros((Bv, Stv, Dv), device=vx.device, dtype=ttok.dtype)
                            for i, ix in enumerate(tidx):
                                Lv = int(ix.numel())
                                if Lv == 0:
                                    continue
                                teach_v[i, :Lv] = ttok[i, ix]
                                tpad[i, :Lv] = False
                                pos_tv[i, :Lv] = pos[0, ix]
                            ctx_tok, ctx_pad = student.encode_context(vx, cmask)
                            # Build pos_tv and cond for validation
                            cond_v = xattn(ctx_tok, ctx_pad, pos_tv)  # [Bv, Stv, Dv]
                            cond_v = cond_v + pos_tv
                            visible_v = (~tpad)
                            if visible_v.any():
                                cond_v_vis = cond_v[visible_v]
                                cond_v_vis = cond_norm(cond_v_vis)
                                cond_v[visible_v] = cond_v_vis
                                yv_flat = teach_v[visible_v]
                                cv_flat = cond_v_vis
                                logp_v = nf.log_prob(yv_flat, cv_flat)
                                vloss = float((-logp_v.mean()).item())
                                val_loss_acc += vloss
                                # Also compute JEPA-style diagnostics on samples for logging
                                with torch.no_grad():
                                    yhat_v_flat = nf.sample(cv_flat)
                                pred_v = torch.zeros_like(teach_v)
                                pred_v[visible_v] = yhat_v_flat
                                align_v, var_v, cov_v = jepa_metrics(pred_v, teach_v, tpad, std_floor=LOSS_STD_FLOOR)
                                val_align_acc += float(align_v.item())
                                val_var_acc += float(var_v.item())
                                val_cov_acc += float(cov_v.item())

                                # --- extra flow viz on val (first batch only) ---
                                if vb_idx == 0:
                                    # pick the first visible target
                                    # cv_flat: (N_vis, D), yhat_v_flat: (N_vis, D), teach_v: (Bv, Stv, D)
                                    cond_sample = cv_flat[0:1]  # (1, D)
                                    teach_sample = teach_v[visible_v][0:1]  # (1, D)
                                    K = 32
                                    cond_rep = cond_sample.repeat(K, 1)
                                    z = torch.randn(K, HIDDEN_DIM, device=cond_rep.device, dtype=cond_rep.dtype)
                                    x_samp, logdet_samp = nf.forward(z, cond_rep)
                                    tb_log_flow_samples_2d(
                                        writer,
                                        tag='val/flow_samples_2d',
                                        step=samples_seen,
                                        flow_samples=x_samp.detach(),
                                        teacher_vec=teach_sample.squeeze(0).detach(),
                                    )
                                    dists = (x_samp - teach_sample).norm(dim=-1)
                                    tb_log_flow_hist(
                                        writer,
                                        tag='val/flow_sample_dists',
                                        step=samples_seen,
                                        dists=dists,
                                    )
                                    cond_norm_val = cond_sample.norm(dim=-1).mean().item()
                                    tb_log_flow_stats(
                                        writer,
                                        step=samples_seen,
                                        cond_norm=cond_norm_val,
                                        logdet=float(logdet_samp.mean().item()),
                                    )

                                n_batches += 1
                    if n_batches > 0:
                        vloss = val_loss_acc / n_batches
                        writer.add_scalar('val/nll', vloss, global_step=samples_seen)
                        writer.add_scalars('loss', {'val': float(vloss)}, global_step=samples_seen)
                        writer.add_scalars('metrics/components_val',
                                           {
                                               'align': val_align_acc / n_batches,
                                               'var': val_var_acc / n_batches,
                                               'cov': val_cov_acc / n_batches,
                                           },
                                           global_step=samples_seen)
                except Exception:
                    pass

                # kNN convergence visualization (fixed projector; stable axes/scale)
                try:
                    val_feats_prev0, tracked_idx = tb_log_knn_convergence(
                        writer, 'viz/knn_convergence', samples_seen,
                        projector2d, tr_feats, val_feats_prev0, va_feats, tracked_idx
                    )
                except Exception:
                    pass

                # Save checkpoints (last + best)
                ckpt_state: Dict[str, Any] = {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'nf': nf.state_dict(),
                    'xattn': xattn.state_dict(),
                    'cond_norm': cond_norm.state_dict(),
                    'opt': opt.state_dict(),
                    'samples_seen': samples_seen,
                    'steps_done': steps_done,
                    'last_log_samples': last_log_samples,
                    'last_val_samples': last_val_samples,
                    'best_knn': best_knn,
                    'hparams': {
                        'PATCH_SIZE': PATCH_SIZE,
                        'HIDDEN_DIM': HIDDEN_DIM,
                        'VIT_LAYERS': VIT_LAYERS,
                        'VIT_HEADS': VIT_HEADS,
                        'NUM_TARGET_BLOCKS': NUM_TARGET_BLOCKS,
                        'LR': LR,
                        'WEIGHT_DECAY': WEIGHT_DECAY,
                    },
                    'rng': {
                        'py': random.getstate(),
                        'np': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    }
                }
                torch.save(ckpt_state, LAST_CKPT)
                print(f"[ckpt] saved last checkpoint → {LAST_CKPT}")
                if knn_top1 > best_knn:
                    best_knn = knn_top1
                    ckpt_state['best_knn'] = best_knn
                    torch.save(ckpt_state, BEST_CKPT)
                    print(f"[ckpt] new best kNN {best_knn:.4f} → {BEST_CKPT}")

        # end epoch

    # Final save
    final_state = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'nf': nf.state_dict(),
        'xattn': xattn.state_dict(),
        'cond_norm': cond_norm.state_dict(),
        'opt': opt.state_dict(),
        'samples_seen': samples_seen,
        'best_knn': best_knn,
    }
    torch.save(final_state, LAST_CKPT)

    writer.flush(); writer.close()

if __name__ == '__main__':
    train_run()
