
"""
Minimal i-JEPA baseline with a Multivariate Mixture Density Network (MDN)
predictor that models a K-component Gaussian mixture over the full target
embedding vector (diagonal covariances).

- Option A (full multivariate MDN)
- K = 5 (mixture components)
- Embedding dim D inferred from ViT hidden dim (HIDDEN_DIM). Default 384.
"""

import os
import math
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

# -----------------
# Hyperparameters
# -----------------
NUM_CLASSES = 100
IMAGE_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 360
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model sizes (inferred embedding dim)
PATCH_SIZE = 8
HIDDEN_DIM = 384   # embedding dim D inferred from this (default 384)
VIT_LAYERS = 10
VIT_HEADS = 8

# JEPA masking / target config
NUM_TARGET_BLOCKS = 4
TARGET_SCALE = (0.15, 0.20)
TARGET_AR = (0.75, 1.50)
MIN_CTX_PATCHES = 8

# EMA for teacher
MOMENTUM_INIT = 0.998
MOMENTUM_FINAL = 1.0

# MDN specific
MDN_K = 5   # mixture components
MDN_MIN_SIGMA = 1e-3

# Checkpoint
CHECKPOINT_DIR = './checkpoints_mdn'
LAST_CKPT = os.path.join(CHECKPOINT_DIR, 'mdn_last.pt')

# Data loader workers
NUM_WORKERS = 2

# ------------------
# Utilities / Dataset
# ------------------
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
        sample = self.data[index]
        image = sample['image']
        label = int(sample.get('label', -1))
        if self.transform:
            image = self.transform(image)
        return image, label

def _filter_label_lt_num_classes(example):
    return int(example.get('label', -1)) < NUM_CLASSES

def get_dataset(split: str) -> TinyImageNetTorch:
    def load_tinyimagenet(_split):
        ds = load_dataset('zh-plus/tiny-imagenet', split=_split)
        ds = ds.filter(_filter_label_lt_num_classes)
        return ds

    # compute mean/std on train once (small dataset)
    train_raw = load_tinyimagenet('train')
    # quick stats (sample-only for speed; fallback deterministic)
    # We will compute mean/std on a subset to keep startup fast
    n_stats = min(2000, len(train_raw))
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in range(n_stats):
        img = train_raw[i]['image']
        t = transforms.ToTensor()(img)
        mean += t.mean(dim=(1,2)).numpy()
        std += t.std(dim=(1,2)).numpy()
    mean /= n_stats
    std /= n_stats

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
    return TinyImageNetTorch(load_tinyimagenet(split), transform=tfms)

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
        t = self.patch_embed(x)                 # [B, D, g, g]
        t = t.flatten(2).transpose(1, 2)        # [B, T, D]
        t = t + self.pos_embed
        for blk in self.transformer_layers:
            t = blk(t, key_padding_mask=None)
        return self.norm(t)                     # [B, T, D]

# -----------------
# Student / Teacher
# -----------------
class Student(nn.Module):
    def __init__(self, backbone: ViTBackbone):
        super().__init__()
        self.backbone = backbone

    def encode_context(self, x: torch.Tensor, context_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        context_mask: [B, T] True = visible context, False = not included
        Returns:
            ctx_out: [B, Sc, D]
            ctx_pad: [B, Sc] True = pad (to be ignored)
        """
        B = x.size(0)
        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)  # [B,T,D]
        pos = self.backbone.pos_embed  # [1,T,D]
        ctx_lists: List[torch.Tensor] = []
        lengths: List[int] = []
        for i in range(B):
            idx = context_mask[i].nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                # ensure at least one context token
                idx = torch.tensor([0], device=tokens.device, dtype=torch.long)
            ti = tokens[i, idx]
            pi = pos[0, idx]
            ctx_lists.append(ti + pi)
            lengths.append(int(idx.numel()))
        Sc = int(max(lengths))
        D = tokens.size(2)
        ctx_pad = torch.ones((B, Sc), dtype=torch.bool, device=tokens.device)
        ctx_out = torch.zeros((B, Sc, D), dtype=tokens.dtype, device=tokens.device)
        for i, ci in enumerate(ctx_lists):
            L = ci.size(0)
            ctx_out[i, :L] = ci
            ctx_pad[i, :L] = False
        # run transformer layers (re-using backbone layers)
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

# ----------------
# Mask utilities
# ----------------
def _sample_blocks(grid: int, n_blocks: int, scale: Tuple[float, float], ar: Tuple[float, float]) -> List[Tuple[int, int, int, int]]:
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
    return mask.flatten()

def sample_context_and_targets(grid: int, min_ctx: int = MIN_CTX_PATCHES, max_tries: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # Fallback
    ctx = torch.ones(grid, grid, dtype=torch.bool)
    tgt_blocks = _sample_blocks(grid, max(1, NUM_TARGET_BLOCKS // 2), (0.10, 0.15), TARGET_AR)
    tgt = torch.zeros(grid, grid, dtype=torch.bool)
    for (yy, xx, hh, ww) in tgt_blocks:
        tgt[yy:yy + hh, xx:xx + ww] = True
    ctx[tgt] = False
    if int(ctx.sum().item()) == 0:
        ctx[0,0] = True; tgt[0,0] = False
    return ctx.flatten(), tgt.flatten()

# ----------------------
# MDN Predictor + Utils
# ----------------------
class MDNPredictor(nn.Module):
    """
    Predictor that attends over [context tokens | target queries] (Transformer),
    then produces MDN parameters (pi, mu, sigma) per-target token.

    Outputs:
        pi:   [B, St, K]   (mixture weights)
        mu:   [B, St, K, D] (means)
        sigma:[B, St, K, D] (std devs, positive)
    """
    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4, K: int = MDN_K):
        super().__init__()
        self.dim = dim
        self.K = K
        self.q_token = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        # MDN heads
        self.pi_head = nn.Linear(dim, K)           # logits for mixture weights
        self.mu_head = nn.Linear(dim, K * dim)     # flattened K x D
        self.sigma_head = nn.Linear(dim, K * dim)  # flattened K x D (will softplus)
        self._init_head(self.pi_head)
        self._init_head(self.mu_head)
        self._init_head(self.sigma_head)

    def _init_head(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, ctx: torch.Tensor, ctx_pad: torch.Tensor, tgt_q: torch.Tensor, tgt_pad: torch.Tensor):
        """
        ctx: [B, Sc, D], ctx_pad: [B, Sc] True=pad
        tgt_q: [B, St, D], tgt_pad: [B, St] True=pad
        returns (pi, mu, sigma) shapes described above
        """
        x = torch.cat([ctx, tgt_q], dim=1)
        pad = torch.cat([ctx_pad, tgt_pad], dim=1)
        for blk in self.layers:
            x = blk(x, key_padding_mask=pad)
        x = self.norm(x)
        Sc = ctx.size(1)
        tgt_repr = x[:, Sc:, :]  # [B, St, D]
        B, St, D = tgt_repr.shape

        # Mixture logits -> pi
        pi_logits = self.pi_head(tgt_repr)  # [B, St, K]
        pi = F.softmax(pi_logits, dim=-1)

        # Means
        mu_flat = self.mu_head(tgt_repr)   # [B, St, K*D]
        mu = mu_flat.view(B, St, self.K, D)

        # Sigma / stddev (positive)
        sigma_flat = self.sigma_head(tgt_repr)
        sigma = F.softplus(sigma_flat).view(B, St, self.K, D)
        sigma = sigma.clamp(min=MDN_MIN_SIGMA)

        return pi, mu, sigma

def mdn_neg_log_likelihood(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, tgt_pad: torch.Tensor) -> torch.Tensor:
    """
    Compute average negative log likelihood under MDN for visible target tokens.

    Shapes:
        pi:   [B, St, K]
        mu:   [B, St, K, D]
        sigma:[B, St, K, D]
        target:[B, St, D]
        tgt_pad: [B, St] True = pad (ignore)
    Returns:
        scalar tensor: mean NLL over visible tokens
    """
    # Expand target to [B, St, K, D]
    B, St, D = target.shape
    K = pi.size(-1)
    target_exp = target.unsqueeze(2)  # [B, St, 1, D]
    # compute log component densities
    # log N(x | mu_k, diag(sigma_k^2)) = -0.5*( D*log(2π) + sum(log(sigma^2)) + sum((x-mu)^2 / sigma^2) )
    var = sigma * sigma  # [B,St,K,D]
    # sum over dims
    # compute log_det_term = -0.5 * sum(log(2π * var), dim=-1)
    log_det_term = -0.5 * torch.sum(torch.log(2 * math.pi * var), dim=-1)  # [B,St,K]
    # compute exponent term
    diff = target_exp - mu  # [B,St,K,D]
    exponent = -0.5 * torch.sum((diff * diff) / var, dim=-1)  # [B,St,K]
    # log component = log(pi) + log_det_term + exponent
    log_pi = torch.log(pi + 1e-12)  # numerical safety
    log_comp = log_pi + log_det_term + exponent  # [B,St,K]
    # log-sum-exp over components
    # stable logsumexp
    max_log, _ = torch.max(log_comp, dim=-1, keepdim=True)  # [B,St,1]
    log_sum = max_log + torch.log(torch.sum(torch.exp(log_comp - max_log), dim=-1, keepdim=True) + 1e-12)  # [B,St,1]
    log_prob = log_sum.squeeze(-1)  # [B,St]
    nll = -log_prob  # [B,St] negative log-likelihood per token
    # mask padding
    visible = (~tgt_pad).float()  # [B,St] 1.0 where valid
    denom = visible.sum()
    if denom == 0.0:
        return torch.tensor(0.0, device=target.device, requires_grad=True)
    loss = (nll * visible).sum() / denom
    return loss

# ----------------
# Training routine
# ----------------
def momentum_schedule(it: int, total_its: int, m0: float = MOMENTUM_INIT, m1: float = MOMENTUM_FINAL) -> float:
    if total_its <= 1:
        return m1
    t = min(1.0, max(0.0, it / (total_its - 1)))
    return m1 - (m1 - m0) * (0.5 * (1.0 + math.cos(math.pi * t)))

def train_run():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Data
    train_ds = get_dataset('train')
    val_ds = get_dataset('valid')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    num_train = len(train_ds)
    print(f"Train samples: {num_train}, Val samples: {len(val_ds)}")

    # Models
    student = Student(ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)).to(DEVICE)
    teacher = Teacher(ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)).to(DEVICE)
    teacher.ema_update(student, m=0.0)
    predictor = MDNPredictor(HIDDEN_DIM, num_layers=2, num_heads=VIT_HEADS, K=MDN_K).to(DEVICE)

    params = list(student.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * max(1, len(train_loader))
    samples_seen = 0
    steps_done = 0

    # Resume if checkpoint exists
    start_epoch = 0
    if os.path.isfile(LAST_CKPT):
        try:
            st = torch.load(LAST_CKPT, map_location=DEVICE)
            student.load_state_dict(st['student'])
            teacher.load_state_dict(st['teacher'])
            predictor.load_state_dict(st['predictor'])
            opt.load_state_dict(st['opt'])
            samples_seen = int(st.get('samples_seen', 0))
            steps_done = int(st.get('steps_done', 0))
            start_epoch = int(st.get('epoch', 0))
            print(f"Resumed checkpoint: samples_seen={samples_seen}, steps_done={steps_done}, epoch={start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    D = HIDDEN_DIM
    print(f"Training on device={DEVICE} | D={D} | MDN_K={MDN_K}")

    for epoch in range(start_epoch, EPOCHS):
        student.train(); predictor.train()
        for batch_idx, (x, _) in enumerate(train_loader, start=1):
            x = x.to(DEVICE)
            B = x.size(0)
            grid = student.backbone.grid
            T = grid * grid

            # sample per-sample masks
            ctx_list, tgt_list = [], []
            for _b in range(B):
                ctx_mask, tgt_mask = sample_context_and_targets(grid)
                ctx_list.append(ctx_mask)
                tgt_list.append(tgt_mask)
            context_mask = torch.stack(ctx_list, dim=0).to(x.device)  # [B,T]
            target_mask = torch.stack(tgt_list, dim=0).to(x.device)   # [B,T]

            # teacher tokens (fixed)
            with torch.no_grad():
                t_tokens = teacher(x)  # [B,T,D]

            # build per-sample target set (pad to St)
            tgt_idxs = [target_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(B)]
            St = int(max([int(idx.numel()) for idx in tgt_idxs]) if tgt_idxs else 0)
            D = t_tokens.size(2)
            teacher_tgt = torch.zeros((B, St, D), device=x.device, dtype=t_tokens.dtype)
            tgt_pad = torch.ones((B, St), dtype=torch.bool, device=x.device)
            pos = student.backbone.pos_embed  # [1,T,D]
            tgt_q = torch.zeros((B, St, D), device=x.device, dtype=t_tokens.dtype)
            for i, idx in enumerate(tgt_idxs):
                L = int(idx.numel())
                if L == 0:
                    continue
                teacher_tgt[i, :L] = t_tokens[i, idx]
                tgt_pad[i, :L] = False
                tgt_q[i, :L] = predictor.q_token[None, :].expand(L, -1) + pos[0, idx]

            # student encodes context only
            ctx_tokens, ctx_pad = student.encode_context(x, context_mask)

            # predictor outputs MDN parameters
            pi, mu, sigma = predictor(ctx_tokens, ctx_pad, tgt_q, tgt_pad)  # shapes: [B,St,K], [B,St,K,D], [B,St,K,D]

            # MDN NLL loss
            mdn_nll = mdn_neg_log_likelihood(pi, mu, sigma, teacher_tgt, tgt_pad)

            # Backprop on student + predictor
            opt.zero_grad(set_to_none=True)
            mdn_nll.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()

            # EMA update for teacher
            samples_seen += B
            steps_done += 1
            m = momentum_schedule(samples_seen, EPOCHS * len(train_loader))
            teacher.ema_update(student, m)

            if batch_idx % 20 == 0:
                print(f"[train] epoch {epoch+1}/{EPOCHS} step {batch_idx}/{len(train_loader)} mdn_nll={mdn_nll.item():.4f}")

        # end epoch; save checkpoint
        ckpt = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'predictor': predictor.state_dict(),
            'opt': opt.state_dict(),
            'samples_seen': samples_seen,
            'steps_done': steps_done,
            'epoch': epoch + 1,
        }
        torch.save(ckpt, LAST_CKPT)
        print(f"[ckpt] saved checkpoint after epoch {epoch+1} -> {LAST_CKPT}")

        # Optional lightweight validation NLL print
        try:
            student.eval(); predictor.eval()
            with torch.no_grad():
                val_losses = []
                for vb_idx, (vx, _) in enumerate(val_loader):
                    if vb_idx >= 5:
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
                    tq = torch.zeros((Bv, Stv, Dv), device=vx.device, dtype=ttok.dtype)
                    for i, ix in enumerate(tidx):
                        Lv = int(ix.numel())
                        if Lv == 0:
                            continue
                        teach_v[i, :Lv] = ttok[i, ix]
                        tpad[i, :Lv] = False
                        tq[i, :Lv] = predictor.q_token[None, :].expand(Lv, -1) + pos[0, ix]
                    ctx_tok, ctx_pad = student.encode_context(vx, cmask)
                    pi_v, mu_v, sigma_v = predictor(ctx_tok, ctx_pad, tq, tpad)
                    vloss = mdn_neg_log_likelihood(pi_v, mu_v, sigma_v, teach_v, tpad)
                    val_losses.append(float(vloss.item()))
                if len(val_losses) > 0:
                    print(f"[val] epoch {epoch+1} avg_mdn_nll={np.mean(val_losses):.4f}")
        except Exception:
            pass

    print("Training finished.")
    # final save
    final_state = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'predictor': predictor.state_dict(),
        'opt': opt.state_dict(),
        'samples_seen': samples_seen,
    }
    torch.save(final_state, LAST_CKPT)

if __name__ == '__main__':
    train_run()
