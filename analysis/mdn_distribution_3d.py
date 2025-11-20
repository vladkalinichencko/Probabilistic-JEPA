#!/usr/bin/env python3
"""
Generate a 3D isometric scatter of MDN samples (projected via PCA).
"""

from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from MDN.mdn import (  # noqa: E402
    ViTBackbone,
    Student,
    Teacher,
    MDNPredictor,
    sample_context_and_targets,
    IMAGE_SIZE,
    PATCH_SIZE,
    HIDDEN_DIM,
    VIT_LAYERS,
    VIT_HEADS,
    MDN_K,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    ckpt_path = ROOT / "MDN" / "mdn_last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("MDN checkpoint not found")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    student = Student(
        ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)
    ).to(DEVICE)
    teacher = Teacher(
        ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)
    ).to(DEVICE)
    predictor = MDNPredictor(HIDDEN_DIM, num_layers=2, num_heads=VIT_HEADS, K=MDN_K).to(
        DEVICE
    )
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    predictor.load_state_dict(ckpt["predictor"])
    student.eval()
    teacher.eval()
    predictor.eval()
    return student, teacher, predictor


def sample_batch(student, teacher, predictor, num_samples=4096):
    dataset = load_dataset("zh-plus/tiny-imagenet", split="valid").filter(
        lambda e: int(e["label"]) < 100
    )
    img = dataset[0]["image"]
    tfm = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    x = tfm(img).unsqueeze(0).to(DEVICE)

    ctx_mask, tgt_mask = sample_context_and_targets(student.backbone.grid)
    ctx_mask = ctx_mask.unsqueeze(0).to(DEVICE)
    tgt_mask = tgt_mask.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        teacher_tokens = teacher(x)
    idx = tgt_mask[0].nonzero(as_tuple=False).squeeze(1)
    St = idx.numel()
    pos = student.backbone.pos_embed
    tgt_q = predictor.q_token[None, :].expand(St, -1) + pos[0, idx]
    tgt_q = tgt_q.unsqueeze(0)
    ctx_tok, ctx_pad = student.encode_context(x, ctx_mask)
    with torch.no_grad():
        pi, mu, sigma = predictor(
            ctx_tok, ctx_pad, tgt_q, torch.zeros((1, St), dtype=torch.bool, device=DEVICE)
        )

    samples = []
    while len(samples) * St < num_samples:
        cat = torch.distributions.Categorical(pi)
        comp_idx = cat.sample().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.size(-1))
        chosen_mu = mu.gather(2, comp_idx).squeeze(2)
        chosen_sigma = sigma.gather(2, comp_idx).squeeze(2)
        draw = torch.normal(chosen_mu, chosen_sigma)
        samples.append(draw)
    samples = torch.cat(samples, dim=1)[:, :num_samples, :]
    return samples[0].cpu().numpy()


def plot_3d(points):
    pca = PCA(n_components=3)
    proj = pca.fit_transform(points)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        proj[:, 0],
        proj[:, 1],
        proj[:, 2],
        s=6,
        c=proj[:, 2],
        cmap="plasma",
        alpha=0.6,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("MDN distribution (3D PCA projection)")
    ax.view_init(elev=25, azim=35)
    out_path = FIG_DIR / "mdn_distribution_3d.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")


def main():
    student, teacher, predictor = load_models()
    points = sample_batch(student, teacher, predictor)
    plot_3d(points)


if __name__ == "__main__":
    main()
