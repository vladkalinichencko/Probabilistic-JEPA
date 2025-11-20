#!/usr/bin/env python3
"""
Compute kNN@1 accuracy for different predictors by extracting backbone
embeddings on Tiny ImageNet-100 (train/val splits).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "analysis" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _import_module(path: Path, module_name: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_embeddings(backbone, loader, device):
    feats = []
    labels = []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            tokens = backbone(x)
            g = tokens.mean(dim=1).cpu()
            feats.append(g)
            labels.append(y.clone())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def _knn_accuracy(train_feats, train_labels, val_feats, val_labels, k=20, chunk_size=256):
    train_norm = F.normalize(train_feats, dim=1)
    val_norm = F.normalize(val_feats, dim=1)
    preds = []
    for start in range(0, val_norm.size(0), chunk_size):
        end = min(val_norm.size(0), start + chunk_size)
        sims = val_norm[start:end] @ train_norm.t()
        topk = sims.topk(k, dim=1).indices
        neighbors = train_labels[topk]
        pred = torch.mode(neighbors, dim=1).values
        preds.append(pred)
    preds = torch.cat(preds)
    acc = (preds == val_labels).float().mean().item()
    return acc * 100.0


def compute_mdn(device):
    module = _import_module(ROOT / "MDN" / "mdn.py", "mdn_knn")
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
    student.load_state_dict(ckpt["student"])
    dataset_train = module.get_dataset("train")
    dataset_val = module.get_dataset("valid")
    train_loader = DataLoader(dataset_train, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=0)
    train_feats, train_labels = _extract_embeddings(student.backbone, train_loader, device)
    val_feats, val_labels = _extract_embeddings(student.backbone, val_loader, device)
    acc = _knn_accuracy(train_feats, train_labels, val_feats, val_labels)
    return acc


def compute_autoregressive(device):
    module = _import_module(ROOT / "autoregression" / "rnade_model.py", "arn_knn")
    ckpt = torch.load(ROOT / "autoregression" / "student_model.pth", map_location=device)
    student = module.Student(
        module.ViTBackbone(
            module.IMAGE_SIZE,
            module.PATCH_SIZE,
            module.HIDDEN_DIM,
            module.VIT_LAYERS,
            module.VIT_HEADS,
        )
    ).to(device)
    student.load_state_dict(ckpt)
    dataset_train = module.get_dataset("train")
    dataset_val = module.get_dataset("valid")
    train_loader = DataLoader(dataset_train, batch_size=128, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=128, shuffle=False, num_workers=0)
    train_feats, train_labels = _extract_embeddings(student.backbone, train_loader, device)
    val_feats, val_labels = _extract_embeddings(student.backbone, val_loader, device)
    acc = _knn_accuracy(train_feats, train_labels, val_feats, val_labels)
    return acc


def compute_baseline(device):
    module = _import_module(ROOT / "I-JEPA" / "i-jepa.py", "ijepa_knn")
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
    student.load_state_dict(ckpt["student"])
    dataset_train = module.get_dataset("train")
    dataset_val = module.get_dataset("valid")
    batch_size = getattr(module, "BATCH_SIZE", 600)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    train_feats, train_labels = _extract_embeddings(student.backbone, train_loader, device)
    val_feats, val_labels = _extract_embeddings(student.backbone, val_loader, device)
    acc = _knn_accuracy(train_feats, train_labels, val_feats, val_labels)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mdn", "autoregressive"],
        choices=["mdn", "autoregressive", "baseline"],
    )
    args = parser.parse_args()
    device = _device()
    results = {}
    for model in args.models:
        if model == "mdn":
            acc = compute_mdn(device)
        elif model == "autoregressive":
            acc = compute_autoregressive(device)
        else:
            acc = compute_baseline(device)
        results[model] = acc
        print(f"{model}: {acc:.2f}%")
    out_path = CACHE_DIR / "knn_metrics.json"
    data = {}
    if out_path.exists():
        data = json.loads(out_path.read_text())
    data.update(results)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved metrics → {out_path}")


if __name__ == "__main__":
    main()
