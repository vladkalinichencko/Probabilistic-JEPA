import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
preprocess = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])
cache = {}


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, name: str = "attention"):
        super().__init__()
        self.attn_name = name
        attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        setattr(self, name, attn)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn = getattr(self, self.attn_name)
        attn_out, _ = attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(self, image_size: int, patch_size: int, hidden_dim: int, num_layers: int, num_heads: int, name: str = "attention"):
        super().__init__()
        assert image_size % patch_size == 0
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        self.transformer_layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, name=name) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        t = t + self.pos_embed
        for blk in self.transformer_layers:
            t = blk(t)
        return self.norm(t)


class Teacher(nn.Module):
    def __init__(self, backbone: ViTBackbone):
        super().__init__()
        self.backbone = backbone
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_teacher(name: str) -> nn.Module:
    if name in cache:
        return cache[name]
    if name == "mdn":
        ckpt = torch.load(ROOT / "MDN" / "mdn_last.pt", map_location=device, weights_only=False)["teacher"]
        backbone = ViTBackbone(64, 8, 384, 10, 8, name="attention")
        teacher = Teacher(backbone)
        teacher.load_state_dict(ckpt)
    elif name == "autoregressive":
        from autoregression import rnade_model as arn

        teacher = arn.Teacher(arn.ViTBackbone(arn.IMAGE_SIZE, arn.PATCH_SIZE, arn.HIDDEN_DIM, arn.VIT_LAYERS, arn.VIT_HEADS))
        teacher.load_state_dict(torch.load(ROOT / "autoregression" / "teacher_model.pth", map_location=device, weights_only=False))
    elif name == "diffusion":
        import diffusion.diffusion as diff

        teacher = diff.Teacher(diff.ViTBackbone(diff.IMAGE_SIZE, diff.PATCH_SIZE, diff.HIDDEN_DIM, diff.VIT_LAYERS, diff.VIT_HEADS))
        teacher.load_state_dict(torch.load(ROOT / "diffusion" / "duffision_weights.pt", map_location=device, weights_only=False)["teacher"])
    elif name == "flow_matching":
        import flow_matching.flow_matching as fm

        ckpt = torch.load(ROOT / "flow_matching" / "flow_matching_last.pt", map_location=device, weights_only=False)
        h = ckpt.get("hparams", {})
        teacher = fm.Teacher(
            fm.ViTBackbone(
                getattr(fm, "IMAGE_SIZE", 64),
                int(h.get("PATCH_SIZE", getattr(fm, "PATCH_SIZE", 8))),
                int(h.get("HIDDEN_DIM", getattr(fm, "HIDDEN_DIM", 384))),
                int(h.get("VIT_LAYERS", getattr(fm, "VIT_LAYERS", 10))),
                int(h.get("VIT_HEADS", getattr(fm, "VIT_HEADS", 8))),
            )
        )
        teacher.load_state_dict(ckpt["teacher"])
    else:
        raise ValueError("Unknown model")
    teacher.to(device).eval()
    cache[name] = teacher
    return teacher


def encode(img: Image.Image, name: str) -> torch.Tensor:
    x = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = load_teacher(name)(x)
        pooled = tokens.mean(dim=1)
    return F.normalize(pooled, dim=1)


def compare(img_a: Image.Image, img_b: Image.Image, name: str) -> str:
    if img_a is None or img_b is None:
        return "Upload two images"
    score = F.cosine_similarity(encode(img_a, name), encode(img_b, name)).item()
    pct = max(min((score + 1) * 50, 100.0), 0.0)
    return f"{pct:.2f}%"


def main():
    st.set_page_config(page_title="Probabilistic JEPA similarity", layout="centered")
    st.title("Probabilistic JEPA similarity")
    st.write("Upload two images, pick a predictor, and see cosine similarity between their latent representations.")

    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Image A", type=["png", "jpg", "jpeg"], key="img_a")
    with col2:
        file_b = st.file_uploader("Image B", type=["png", "jpg", "jpeg"], key="img_b")

    model_name = st.selectbox("Predictor", ["mdn", "autoregressive", "diffusion", "flow_matching"], index=0)

    if st.button("Compute similarity"):
        if not file_a or not file_b:
            st.warning("Please upload both images.")
        else:
            try:
                img_a = Image.open(file_a)
                img_b = Image.open(file_b)
                result = compare(img_a, img_b, model_name)
                st.success(f"Similarity: {result}")
                st.image([img_a.resize((128, 128)), img_b.resize((128, 128))], caption=["A", "B"])
            except Exception as e:
                st.error(f"Failed to compute similarity: {e}")


if __name__ == "__main__":
    main()
