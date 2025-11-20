#!/usr/bin/env python3
"""
Utility classes and helpers for the probabilistic autoregressive (RNADE-style)
predictor.  Extracted from the original notebook so that training, evaluation,
and visualization scripts can import a plain Python module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# -----------------
# Hyperparameters
# -----------------
NUM_CLASSES = 100
IMAGE_SIZE = 64
PATCH_SIZE = 8
HIDDEN_DIM = 600
VIT_LAYERS = 6
VIT_HEADS = 6
BATCH_SIZE = 128
NUM_COMPONENTS = 4
EMA_MOMENTUM = 0.996
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


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
        label = int(self.data[index]["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


def _filter_label_lt_num_classes(example):
    return int(example.get("label", -1)) < NUM_CLASSES


def get_dataset(split: str) -> TinyImageNetTorch:
    def load_tinyimagenet(_split):
        ds = load_dataset("zh-plus/tiny-imagenet", split=_split)
        ds = ds.filter(_filter_label_lt_num_classes)
        return ds

    train_raw = load_tinyimagenet("train")
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = len(train_raw)
    for sample in train_raw:
        img = transforms.ToTensor()(sample["image"])
        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))
    mean /= n_samples
    std /= n_samples

    raw_dataset = load_tinyimagenet(split)
    if split == "train":
        tfms = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist()),
            ]
        )
    else:
        tfms = transforms.Compose(
            [
                ToRGB(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist()),
            ]
        )
    return TinyImageNetTorch(raw_dataset, transform=tfms)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(self, image_size, patch_size, hidden_dim, num_layers, num_heads):
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

    def forward(self, x):
        t = self.patch_embed(x)
        t = t.flatten(2).transpose(1, 2)
        t = t + self.pos_embed
        for blk in self.transformer_layers:
            t = blk(t)
        return self.norm(t)


class Student(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def encode_context(self, x, context_mask):
        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        pos = self.backbone.pos_embed
        ctx_lists, lengths = [], []
        for i in range(tokens.size(0)):
            idx = context_mask[i].nonzero(as_tuple=False).squeeze(1)
            ti = tokens[i, idx]
            pi = pos[0, idx]
            ctx_lists.append(ti + pi)
            lengths.append(int(idx.numel()))
        Sc = int(max(lengths)) if lengths else 0
        D = tokens.size(2)
        ctx_pad = torch.ones((x.size(0), Sc), dtype=torch.bool, device=x.device)
        ctx_out = torch.zeros((x.size(0), Sc, D), dtype=tokens.dtype, device=x.device)
        for i, ci in enumerate(ctx_lists):
            L = ci.size(0)
            ctx_out[i, :L] = ci
            ctx_pad[i, :L] = False
        for blk in self.backbone.transformer_layers:
            ctx_out = blk(ctx_out, key_padding_mask=ctx_pad)
        ctx_out = self.backbone.norm(ctx_out)
        return ctx_out, ctx_pad


class Teacher(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def ema_update(self, student, m=EMA_MOMENTUM):
        for p_t, p_s in zip(self.backbone.parameters(), student.backbone.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))


class ActNorm1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x: torch.Tensor):
        flat = x.reshape(-1, self.dim)
        if flat.numel() == 0:
            return
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False).clamp(min=self.eps)
        with torch.no_grad():
            self.bias.copy_(-mean)
            self.log_scale.copy_(torch.log(1.0 / std))
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.initialized.item() == 0 and self.training:
            self.initialize(x.detach())
        scale = torch.exp(self.log_scale)
        return (x + self.bias) * scale


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)


def create_masks(input_dim: int, hidden_dims: Sequence[int], num_components: int):
    D = input_dim
    layers = [D] + list(hidden_dims) + [D * num_components * 3]
    rng = torch.Generator().manual_seed(0)
    degrees = []
    for i, layer_dim in enumerate(layers):
        if i == 0:
            degrees.append(torch.arange(1, D + 1))
        else:
            min_degree = torch.min(degrees[-1]).item()
            degrees.append(torch.randint(min_degree, D + 1, (layer_dim,), generator=rng))
    masks = []
    for i in range(len(layers) - 1):
        mask = (degrees[i + 1].unsqueeze(-1) >= degrees[i].unsqueeze(0)).float()
        masks.append(mask)
    return masks


class MaskedAutoregressiveMLPConditional(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_components, context_dim):
        super().__init__()
        self.input_dim = input_dim
        self.num_components = num_components
        masks = create_masks(input_dim, hidden_dims, num_components)
        self.net = nn.ModuleList()
        self.hidden_acts = nn.ModuleList([nn.ReLU() for _ in hidden_dims])
        self.film_scale = nn.ModuleList()
        self.film_shift = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [input_dim * num_components * 3]
        self.act_norms = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.net.append(MaskedLinear(dims[i], dims[i + 1], masks[i]))
            if i < len(hidden_dims):
                self.film_scale.append(nn.Linear(context_dim, dims[i + 1]))
                self.film_shift.append(nn.Linear(context_dim, dims[i + 1]))
                self.act_norms.append(ActNorm1D(dims[i + 1]))

    def forward(self, x, context):
        h = x
        for i, layer in enumerate(self.net[:-1]):
            h = layer(h)
            s = self.film_scale[i](context) * 0.1
            b = torch.tanh(self.film_shift[i](context))
            h = h * (1 + s) + b
            h = self.hidden_acts[i](h)
        out = self.net[-1](h)
        D, K = self.input_dim, self.num_components
        out = out.view(-1, D, K, 3)
        alpha_raw = out[..., 0]
        temperature = 2.0
        alpha = F.softmax(alpha_raw / temperature, dim=-1)
        eps = 0.2
        alpha = (1 - eps) * alpha + eps / K
        mu = out[..., 1]
        log_sigma = torch.clamp(out[..., 2], min=-10.0, max=10.0)
        sigma = F.softplus(log_sigma) + 1e-4
        return alpha, mu, sigma


def _sample_blocks(grid, n_blocks, scale, ar):
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


def build_mask_from_blocks(grid, blocks):
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    for (y0, x0, h, w) in blocks:
        mask[y0 : y0 + h, x0 : x0 + w] = True
    return mask.flatten()


def sample_context_and_targets(grid, min_ctx=8, max_tries=50):
    for _ in range(max_tries):
        ctx_block = _sample_blocks(grid, 1, (0.85, 1.0), (0.75, 1.5))[0]
        tgt_blocks = _sample_blocks(grid, 4, (0.15, 0.20), (0.75, 1.5))
        ctx = torch.zeros(grid, grid, dtype=torch.bool)
        y0, x0, h, w = ctx_block
        ctx[y0 : y0 + h, x0 : x0 + w] = True
        tgt = torch.zeros(grid, grid, dtype=torch.bool)
        for (yy, xx, hh, ww) in tgt_blocks:
            tgt[yy : yy + hh, xx : xx + ww] = True
        ctx[tgt] = False
        if int(ctx.sum().item()) >= min_ctx and int(tgt.sum().item()) > 0:
            return ctx.flatten(), tgt.flatten()
    ctx = torch.ones(grid, grid, dtype=torch.bool)
    tgt_blocks = _sample_blocks(grid, 2, (0.10, 0.15), (0.75, 1.5))
    tgt = torch.zeros(grid, grid, dtype=torch.bool)
    for (yy, xx, hh, ww) in tgt_blocks:
        tgt[yy : yy + hh, xx : xx + ww] = True
    ctx[tgt] = False
    if int(ctx.sum().item()) == 0:
        ctx[0, 0] = True
        tgt[0, 0] = False
    return ctx.flatten(), tgt.flatten()


@torch.no_grad()
def extract_global_embeddings(backbone: ViTBackbone, loader: DataLoader, device: torch.device):
    backbone.eval()
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        tokens = backbone(x)
        g = tokens.mean(dim=1)
        feats.append(g.cpu())
        labels.append(y.clone())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def knn_eval(train_feats, train_labels, val_feats, val_labels, k=20):
    train_norm = F.normalize(train_feats, dim=1)
    val_norm = F.normalize(val_feats, dim=1)
    sim = val_norm @ train_norm.t()
    topk = sim.topk(k, dim=1).indices
    neighbors = train_labels[topk]
    preds = torch.mode(neighbors, dim=1).values
    acc = (preds == val_labels).float().mean().item()
    return float(acc)


def mixture_gaussian_nll_loss(target, alpha, mu, sigma):
    B, D = target.shape
    target = target.unsqueeze(-1)
    log_prob = (
        -torch.log(sigma + 1e-8)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((target - mu) / (sigma + 1e-8)) ** 2
    )
    log_prob = torch.log(alpha + 1e-8) + log_prob
    log_prob = torch.logsumexp(log_prob, dim=-1)
    nll = -log_prob.mean()
    return nll


@dataclass
class CheckpointPaths:
    student: str
    predictor: str
    teacher: str

