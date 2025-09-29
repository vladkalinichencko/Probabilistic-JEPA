from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def _to_uint8_rgb(img3chw: torch.Tensor) -> np.ndarray:
    x = img3chw.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    x = (x * 255.0).clamp(0, 255).byte().numpy()
    return np.transpose(x, (1, 2, 0))


class ModelInterpreter:
    """Encapsulates model interpretability visualizations (CNN feature maps, Grad-CAM, saliency, ViT attention)."""

    def __init__(self, image_size: int, device: torch.device):
        self.image_size = image_size
        self.device = device

    def feature_maps_cnn(self, enc: nn.Module, x: torch.Tensor, n_maps: int = 8, title: str = "Feature Maps") -> go.Figure:
        target_conv = None
        for m in reversed(list(enc.features.modules())):
            if isinstance(m, nn.Conv2d):
                target_conv = m
                break
        assert target_conv is not None, "No Conv2d layer found in encoder.features"

        acts = {}
        def fwd_hook(module, inp, out):
            acts['feat'] = out.detach()
        h = target_conv.register_forward_hook(fwd_hook)
        was_training = enc.training
        enc.eval()
        with torch.no_grad():
            _ = enc(x.to(self.device))
        if was_training:
            enc.train()
        h.remove()

        feat = acts['feat'][0]
        c = min(n_maps, feat.size(0))
        maps = feat[:c]
        maps = (maps - maps.amin(dim=(1,2), keepdim=True)) / (maps.amax(dim=(1,2), keepdim=True) - maps.amin(dim=(1,2), keepdim=True) + 1e-6)

        rows = int(np.ceil(c / 4))
        cols = min(4, c)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"ch {i}" for i in range(c)], vertical_spacing=0.08)
        for i in range(c):
            rr = i // cols + 1
            cc = i % cols + 1
            z = maps[i].detach().cpu().numpy()
            fig.add_trace(go.Heatmap(z=z, colorscale='Inferno', showscale=False), row=rr, col=cc)
        for i in range(rows*cols):
            rr = i // cols + 1
            cc = i % cols + 1
            fig.update_xaxes(visible=False, row=rr, col=cc)
            fig.update_yaxes(visible=False, row=rr, col=cc)
        fig.update_layout(height=max(400, rows*220), width=900, title=title, showlegend=False)
        return fig

    def gradcam_cnn(self, enc: nn.Module, clf: nn.Module, x: torch.Tensor, target_class: Optional[int] = None, alpha: float = 0.45, title: str = "Grad-CAM") -> go.Figure:
        target_conv = None
        for m in reversed(list(enc.features.modules())):
            if isinstance(m, nn.Conv2d):
                target_conv = m
                break
        assert target_conv is not None, "No Conv2d layer found in encoder.features"

        feats, grads = {}, {}
        def fwd_hook(module, inp, out):
            feats['v'] = out
        def bwd_hook(module, gin, gout):
            grads['v'] = gout[0]
        h1 = target_conv.register_forward_hook(fwd_hook)
        h2 = target_conv.register_full_backward_hook(bwd_hook)

        model = nn.Sequential(enc, clf).to(self.device)
        model.eval()

        x = x.to(self.device).requires_grad_(True)
        logits = model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad()
        logits[0, target_class].backward()

        A = feats['v'][0]
        dA = grads['v'][0]
        w = dA.mean(dim=(1,2))
        cam = (w[:, None, None] * A).sum(dim=0)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        cam = torch.nn.functional.interpolate(cam[None, None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0,0]

        img_rgb = _to_uint8_rgb(x[0].detach())

        h1.remove(); h2.remove()

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Image(z=img_rgb))
        fig.add_trace(go.Heatmap(z=cam.detach().cpu().numpy(), colorscale='Jet', opacity=alpha, showscale=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, width=420, title=title)
        return fig

    def saliency(self, model: nn.Module, x: torch.Tensor, target_class: Optional[int] = None, alpha: float = 0.45, title: str = "Saliency") -> go.Figure:
        model = model.to(self.device)
        model.eval()
        x = x.to(self.device).requires_grad_(True)
        logits = model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad()
        logits[0, target_class].backward()

        g = x.grad.detach()[0]
        sal = g.abs().amax(dim=0)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)

        img_rgb = _to_uint8_rgb(x[0].detach())
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Image(z=img_rgb))
        fig.add_trace(go.Heatmap(z=sal.cpu().numpy(), colorscale='Viridis', opacity=alpha, showscale=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, width=420, title=title)
        return fig

    def vit_attention(self, vit: nn.Module, model: nn.Module, x: torch.Tensor, layer: int = -1, alpha: float = 0.5, title: str = "ViT CLS Attention") -> go.Figure:
        attn_store = []
        layers = vit.transformer_layers
        idx = layer if layer >= 0 else (len(layers) + layer)
        idx = max(0, min(idx, len(layers)-1))
        blk = layers[idx]

        def attn_hook(module, inp, out):
            attn_store.append(out[1].detach())
        h = blk.attention.register_forward_hook(attn_hook)

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            _ = model(x.to(self.device))
        h.remove()

        assert len(attn_store) > 0, "No attention captured; ensure forward ran."
        A = attn_store[0][0]
        cls_to_all = A[0]
        patch_scores = cls_to_all[1:]
        grid = int(np.sqrt(vit.num_patches))
        attn_map = patch_scores[:grid*grid].reshape(grid, grid)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
        attn_up = torch.nn.functional.interpolate(attn_map[None,None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0,0]

        img_rgb = _to_uint8_rgb(x[0].detach())
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Image(z=img_rgb))
        fig.add_trace(go.Heatmap(z=attn_up.cpu().numpy(), colorscale='Viridis', opacity=alpha, showscale=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=420, width=420, title=title)
        return fig


class TrainingVisualizer:
    """Tracks training/validation metrics and builds plots."""

    def __init__(
        self,
        device: torch.device,
        capture_gradients: bool = True,
        grad_sample_size: int = 8192,
        num_classes: int = 100,
        logger=None,
        interpreter: ModelInterpreter = None,
        enc: nn.Module = None,
        clf: nn.Module = None,
        encoder_type: str = 'conv',
        vis_sample: Optional[torch.Tensor] = None,
        plot_every_steps: int = 10,
        scalar_every_steps: int = 10,
        heavy_plots_on_validation_only: bool = True,
    ):
        self.device = device
        self.capture_gradients = capture_gradients
        self.grad_sample_size = grad_sample_size
        self.num_classes = num_classes
        self.logger = logger

        # interpretability
        self.interpreter = interpreter
        self.enc = enc
        self.clf = clf
        self.encoder_type = encoder_type
        self.vis_sample = vis_sample
        self.plot_every_steps = max(1, int(plot_every_steps))
        self.scalar_every_steps = max(1, int(scalar_every_steps))
        self.heavy_plots_on_validation_only = bool(heavy_plots_on_validation_only)

        # logs
        self.train_steps: List[Dict] = []
        self.val_steps: List[Dict] = []
        self._grad_snapshots: List[np.ndarray] = []
        self._last_confusion: Optional[Tuple[int, np.ndarray]] = None  # (step, cm)
        self._cm_work: Optional[np.ndarray] = None

        # for update/weight ratio
        self._prev_params: Optional[List[torch.Tensor]] = None

        # gradient noise proxy via Welford on grad_norm
        self._noise_n = 0
        self._noise_mean = 0.0
        self._noise_M2 = 0.0

    def _flatten_grad_snapshot(self, model: nn.Module) -> np.ndarray:
        if not self.capture_gradients:
            return np.array([])
        grads = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().flatten()
            grads.append(g)
        if not grads:
            return np.array([])
        v = torch.cat(grads)
        # Downsample to fixed size deterministically for memory/plotting
        if v.numel() > self.grad_sample_size:
            idx = torch.linspace(0, v.numel()-1, steps=self.grad_sample_size, device=v.device).long()
            v = v[idx]
        return v.cpu().numpy()

    def before_step(self, model: nn.Module) -> None:
        # snapshot parameters pre-optimizer step for update/weight ratio
        self._prev_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    def _compute_update_weight_ratio(self, model: nn.Module) -> float:
        if not self._prev_params:
            return float('nan')
        num_sq = 0.0
        den_sq = 0.0
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            prev = self._prev_params[i]
            i += 1
            d = (p.detach() - prev).float()
            num_sq += float((d*d).sum().item())
            den_sq += float((prev.float()*prev.float()).sum().item())
        self._prev_params = None
        if den_sq <= 0.0:
            return float('inf') if num_sq > 0 else 0.0
        return float(np.sqrt(num_sq) / (np.sqrt(den_sq) + 1e-12))

    def _update_noise(self, grad_norm: float) -> float:
        # Welford update for running std of grad_norm
        self._noise_n += 1
        delta = grad_norm - self._noise_mean
        self._noise_mean += delta / self._noise_n
        delta2 = grad_norm - self._noise_mean
        self._noise_M2 += delta * delta2
        if self._noise_n < 2:
            return 0.0
        var = self._noise_M2 / (self._noise_n - 1)
        std = float(np.sqrt(max(0.0, var)))
        return std / (abs(self._noise_mean) + 1e-12)

    def log_step(self, *, model: nn.Module, epoch: int, step: int, batch_loss: float, batch_acc: float, lr: float, grad_norm: float) -> None:
        update_ratio = self._compute_update_weight_ratio(model)
        grad_noise = self._update_noise(grad_norm)
        self.train_steps.append({
            'epoch': epoch,
            'step': step,
            'loss': float(batch_loss),
            'acc': float(batch_acc),
            'lr': float(lr),
            'grad_norm': float(grad_norm),
            'update_ratio': float(update_ratio),
            'grad_noise': float(grad_noise),
        })
        snap = self._flatten_grad_snapshot(model)
        if snap.size > 0:
            self._grad_snapshots.append(snap)
    def begin_validation(self) -> None:
        self._cm_work = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update_confusion_batch(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if self._cm_work is None:
            self.begin_validation()
        preds = preds.detach().view(-1).cpu().numpy()
        targets = targets.detach().view(-1).cpu().numpy()
        for t, p in zip(targets, preds):
            self._cm_work[int(t), int(p)] += 1

    def end_validation(self, *, step: int, val_loss: float, val_acc: float) -> None:
        self.val_steps.append({
            'step': step,
            'loss': float(val_loss),
            'acc': float(val_acc),
        })
        if self._cm_work is not None:
            self._last_confusion = (int(step), self._cm_work.copy())
            self._cm_work = None
            # (external logging handled by callers)

    def plot_curves(self) -> Tuple[go.Figure, go.Figure]:
        # Loss figure (train vs val in one plot)
        steps = [d['step'] for d in self.train_steps]
        train_loss = [d['loss'] for d in self.train_steps]
        vsteps = [d['step'] for d in self.val_steps]
        val_loss = [d['loss'] for d in self.val_steps]

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=steps, y=train_loss, mode='lines', name='train_loss'))
        loss_fig.add_trace(go.Scatter(x=vsteps, y=val_loss, mode='lines+markers', name='val_loss'))
        loss_fig.update_layout(title='Loss (train vs val)', xaxis_title='step', yaxis_title='loss')

        # Accuracy figure (train vs val in one plot)
        train_acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.train_steps]
        val_acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.val_steps]
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(x=steps, y=train_acc, mode='lines', name='train_acc'))
        acc_fig.add_trace(go.Scatter(x=vsteps, y=val_acc, mode='lines+markers', name='val_acc'))
        acc_fig.update_layout(title='Accuracy (train vs val)', xaxis_title='step', yaxis_title='accuracy', yaxis=dict(range=[0, 1]))

        return loss_fig, acc_fig

    def plot_train_only(self) -> Tuple[go.Figure, go.Figure]:
        steps = [d['step'] for d in self.train_steps]
        loss = [d['loss'] for d in self.train_steps]
        acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.train_steps]
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=steps, y=loss, mode='lines', name='train_loss'))
        fig_l.update_layout(title='Train Loss', xaxis_title='step', yaxis_title='loss')
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=steps, y=acc, mode='lines', name='train_acc'))
        fig_a.update_layout(title='Train Accuracy', xaxis_title='step', yaxis_title='accuracy', yaxis=dict(range=[0,1]))
        return fig_l, fig_a

    def plot_val_only(self) -> Tuple[go.Figure, go.Figure]:
        steps = [d['step'] for d in self.val_steps]
        loss = [d['loss'] for d in self.val_steps]
        acc = [max(0.0, min(1.0, float(d['acc']))) for d in self.val_steps]
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=steps, y=loss, mode='lines+markers', name='val_loss'))
        fig_l.update_layout(title='Validation Loss', xaxis_title='step', yaxis_title='loss')
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=steps, y=acc, mode='lines+markers', name='val_acc'))
        fig_a.update_layout(title='Validation Accuracy', xaxis_title='step', yaxis_title='accuracy', yaxis=dict(range=[0,1]))
        return fig_l, fig_a

    def plot_grad_norm(self) -> go.Figure:
        steps = [d['step'] for d in self.train_steps]
        norms = [d['grad_norm'] for d in self.train_steps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=norms, mode='lines', name='grad_norm'))
        fig.update_layout(title='Gradient Norm over Steps', xaxis_title='step', yaxis_title='||grad||')
        return fig

    def plot_update_ratio(self) -> go.Figure:
        steps = [d['step'] for d in self.train_steps]
        ratios = [d['update_ratio'] for d in self.train_steps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=ratios, mode='lines', name='update/weight'))
        fig.update_layout(title='Update/Weight Ratio', xaxis_title='step', yaxis_title='||Δw||/||w||')
        return fig

    def plot_grad_noise(self) -> go.Figure:
        steps = [d['step'] for d in self.train_steps]
        noise = [d['grad_noise'] for d in self.train_steps]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=noise, mode='lines', name='grad_noise_index'))
        fig.update_layout(title='Gradient Noise Index (std/mean of ||grad||)', xaxis_title='step', yaxis_title='std/mean')
        return fig

    def plot_training_landscape_3d(self) -> Optional[go.Figure]:
        if len(self._grad_snapshots) < 3:
            return None
        G = np.stack(self._grad_snapshots, axis=0)  # [T, D]
        # Center
        Gc = G - G.mean(axis=0, keepdims=True)
        # PCA via SVD, take first 2 PCs
        try:
            U, S, Vt = np.linalg.svd(Gc, full_matrices=False)
            PCs = Vt[:2].T  # [D, 2]
            XY = Gc @ PCs     # [T, 2]
        except Exception:
            # Fallback: random projection
            rng = np.random.default_rng(0)
            R = rng.standard_normal(size=(G.shape[1], 2))
            XY = Gc @ R

        steps = [d['step'] for d in self.train_steps[:XY.shape[0]]]
        losses = [d['loss'] for d in self.train_steps[:XY.shape[0]]]
        xs, ys = XY[:, 0], XY[:, 1]

        # Build a coarse surface via bin-averaged losses
        nx, ny = 40, 40
        # Pad ranges for readability
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_pad = max(1e-6, 0.2 * (x_max - x_min))
        y_pad = max(1e-6, 0.2 * (y_max - y_min))
        x_edges = np.linspace(x_min - x_pad, x_max + x_pad, nx + 1)
        y_edges = np.linspace(y_min - y_pad, y_max + y_pad, ny + 1)
        counts, _, _ = np.histogram2d(xs, ys, bins=(x_edges, y_edges))
        sums, _, _ = np.histogram2d(xs, ys, bins=(x_edges, y_edges), weights=np.array(losses))
        Z = sums / (counts + 1e-12)
        Z[counts < 1] = np.nan
        Xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        Yc = 0.5 * (y_edges[:-1] + y_edges[1:])

        surface = go.Surface(x=Xc, y=Yc, z=Z.T, colorscale='Viridis', opacity=0.65, showscale=True, colorbar=dict(title='loss'))
        path = go.Scatter3d(
            x=xs, y=ys, z=losses,
            mode='lines+markers',
            marker=dict(size=2, color='blue', opacity=0.8),
            line=dict(width=6, color='rgba(30, 90, 200, 0.9)')
        )

        start = go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[losses[0]], mode='markers', marker=dict(size=6, color='green'), name='start')
        end = go.Scatter3d(x=[xs[-1]], y=[ys[-1]], z=[losses[-1]], mode='markers', marker=dict(size=6, color='red'), name='end')

        fig = go.Figure(data=[surface, path, start, end])
        fig.update_layout(
            title='Gradient-derived Training Landscape (surface + path)',
            width=1600, height=600,
            scene=dict(
                xaxis_title='PC1(grad)', yaxis_title='PC2(grad)', zaxis_title='loss',
                xaxis=dict(range=[x_edges[0], x_edges[-1]]),
                yaxis=dict(range=[y_edges[0], y_edges[-1]]),
                aspectmode='manual',
                aspectratio=dict(x=5, y=1, z=1),
                camera=dict(eye=dict(x=2.2, y=1.4, z=1.2))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    def plot_confusion_matrix(self, normalize: bool = True) -> Optional[go.Figure]:
        if not self._last_confusion:
            return None
        step, cm = self._last_confusion
        cm_plot = cm.astype(np.float64)
        if normalize:
            row_sums = cm_plot.sum(axis=1, keepdims=True) + 1e-12
            cm_plot = cm_plot / row_sums
        fig = go.Figure(data=go.Heatmap(z=cm_plot, colorscale='Blues', colorbar=dict(title='freq')))
        fig.update_layout(title=f'Confusion Matrix (step {step})', xaxis_title='Pred', yaxis_title='True')
        return fig

    def end_of_training(self, show: bool = True, report: bool = True) -> None:
        # Build plots (reduced set)
        loss_fig, acc_fig = self.plot_curves()
        grad_fig = self.plot_grad_norm()
        land_fig = self.plot_training_landscape_3d()
        cm_fig = self.plot_confusion_matrix(normalize=True)

        # Show locally
        if show:
            try:
                loss_fig.show(); acc_fig.show(); grad_fig.show()
                if land_fig is not None:
                    land_fig.show()
                if cm_fig is not None:
                    cm_fig.show()
            except Exception:
                pass
        # (external logging handled by callers)

    def overfitting_indicator(self) -> Dict[str, float]:
        # Simple diagnostics: last val - train gap and trend
        if not self.val_steps:
            return {}
        # Align by nearest step
        v = self.val_steps[-1]
        # Find closest train step
        t = min(self.train_steps, key=lambda d: abs(d['step'] - v['step']))
        gap_loss = v['loss'] - t['loss']
        gap_acc = t['acc'] - v['acc']
        return {
            'loss_gap_last': float(gap_loss),
            'acc_gap_last': float(gap_acc),
        }


# --- TensorBoard helpers ---
def _fig_to_uint8_hwc(fig: go.Figure, width: int = 900, height: int = 600) -> Optional[np.ndarray]:
    try:
        # plotly -> PNG bytes requires kaleido; wrap bytes with BytesIO for image reader
        png_bytes = pio.to_image(fig, format='png', width=width, height=height, scale=1)
        import io
        import imageio
        buf = io.BytesIO(png_bytes)
        img = imageio.v2.imread(buf)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        return img
    except Exception:
        return None


def _tb_add_figure(writer: 'SummaryWriter', tag: str, fig: go.Figure, step: int) -> None:
    if writer is None:
        return
    img = _fig_to_uint8_hwc(fig)
    if img is None:
        return
    chw = np.transpose(img, (2,0,1))
    try:
        writer.add_image(tag, chw, global_step=step)
    except Exception:
        pass


def compute_diag_metrics(pred_tgt: torch.Tensor, teacher_tgt: torch.Tensor, tgt_pad: torch.Tensor,
                         context_mask: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, float]:
    mask = (~tgt_pad).float()
    diff = (pred_tgt - teacher_tgt).pow(2).sum(dim=-1) * mask
    mse = (diff.sum() / (mask.sum() + 1e-12)).item()
    pred_norm = (pred_tgt.norm(dim=-1) * mask).sum() / (mask.sum() + 1e-12)
    teach_norm = (teacher_tgt.norm(dim=-1) * mask).sum() / (mask.sum() + 1e-12)
    cos = torch.nn.functional.cosine_similarity(pred_tgt, teacher_tgt, dim=-1)
    cos = (cos * mask).sum() / (mask.sum() + 1e-12)
    T = context_mask.size(1)
    ctx_ratio = context_mask.float().sum(dim=1).mean().item() / float(T)
    tgt_ratio = target_mask.float().sum(dim=1).mean().item() / float(T)
    return {
        'train/token_mse': float(mse),
        'train/pred_norm': float(pred_norm.item()),
        'train/teach_norm': float(teach_norm.item()),
        'train/cos': float(cos.item()),
        'train/ctx_ratio': float(ctx_ratio),
        'train/tgt_ratio': float(tgt_ratio),
    }


def tb_log_scalars(writer: 'SummaryWriter', step: int, scalars: Dict[str, float]) -> None:
    if writer is None:
        return
    for k, v in scalars.items():
        try:
            writer.add_scalar(k, float(v), global_step=step)
        except Exception:
            pass


def tb_log_mask_overlay(writer: 'SummaryWriter', tag: str, step: int,
                        x: torch.Tensor, context_mask: torch.Tensor, target_mask: torch.Tensor,
                        grid: int) -> None:
    try:
        img = x[0].detach().cpu()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).clamp(0,255).byte().numpy()
        img = np.transpose(img, (1,2,0))
        H, W, _ = img.shape
        cell_h, cell_w = H // grid, W // grid
        overlay = img.copy()
        cm = context_mask[0].view(grid, grid).cpu().numpy().astype(bool)
        tm = target_mask[0].view(grid, grid).cpu().numpy().astype(bool)
        for yy in range(grid):
            for xx in range(grid):
                y0, x0 = yy*cell_h, xx*cell_w
                y1, x1 = y0+cell_h, x0+cell_w
                if tm[yy, xx]:
                    overlay[y0:y1, x0:x1, 0] = 255
                    overlay[y0:y1, x0:x1, 1:] = (0.7*overlay[y0:y1, x0:x1, 1:]).astype(np.uint8)
                elif cm[yy, xx]:
                    overlay[y0:y1, x0:x1, 1] = 255
                    overlay[y0:y1, x0:x1, (0,2)] = (0.7*overlay[y0:y1, x0:x1, (0,2)]).astype(np.uint8)
        chw = np.transpose(overlay, (2,0,1))
        writer.add_image(tag, chw, global_step=step)
    except Exception:
        pass


def tb_log_vector_projection(writer: 'SummaryWriter', tag: str, step: int,
                              pred_tgt: torch.Tensor, teacher_tgt: torch.Tensor, tgt_pad: torch.Tensor) -> None:
    try:
        mask = (~tgt_pad)
        P = pred_tgt[mask].detach().cpu().numpy()
        T = teacher_tgt[mask].detach().cpu().numpy()
        if P.size == 0 or T.size == 0:
            return
        X = np.concatenate([T, P], axis=0)
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        PCs = Vt[:2].T
        ZT = (T - X.mean(0)) @ PCs
        ZP = (P - X.mean(0)) @ PCs
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,4), dpi=150)
        ax.scatter(ZT[:,0], ZT[:,1], s=8, c='tab:blue', alpha=0.6, label='teacher')
        ax.scatter(ZP[:,0], ZP[:,1], s=8, c='tab:orange', alpha=0.6, label='pred')
        ax.legend(frameon=False)
        ax.set_title('Target vectors (PCA)')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        import io
        buf = io.BytesIO()
        fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig)
        buf.seek(0)
        import imageio
        img = imageio.v2.imread(buf)
        chw = np.transpose(img, (2,0,1))
        writer.add_image(tag, chw, global_step=step)
    except Exception:
        pass


class FixedProjector2D:
    """Fixed 2D projector for consistent plots across time.

    - Fits PCA(2) on a reference set once (e.g., first train feature bank)
    - Stores mean, basis, and fixed axis limits (percentile-based) for stability
    - Projects any future features into same 2D coordinates and range
    """

    def __init__(self, perc: float = 99.5, seed: int = 42):
        self.perc = float(perc)
        self.seed = int(seed)
        self.mean_: Optional[np.ndarray] = None
        self.P_: Optional[np.ndarray] = None  # [D,2]
        self.xlim_: Optional[Tuple[float,float]] = None
        self.ylim_: Optional[Tuple[float,float]] = None

    def fit(self, X: torch.Tensor) -> 'FixedProjector2D':
        Xn = X.detach().cpu().numpy()
        mu = Xn.mean(axis=0, keepdims=True)
        Xc = Xn - mu
        # PCA via SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:2].T  # [D,2]
        Z = Xc @ P
        # robust limits
        low = (100 - self.perc) / 2.0
        high = 100 - low
        xlim = np.percentile(Z[:,0], [low, high]).tolist()
        ylim = np.percentile(Z[:,1], [low, high]).tolist()
        # small padding
        def pad(lim):
            a, b = float(lim[0]), float(lim[1])
            m = (b - a) * 0.05 + 1e-8
            return (a - m, b + m)
        self.mean_ = mu[0]
        self.P_ = P
        self.xlim_ = pad(xlim)
        self.ylim_ = pad(ylim)
        return self

    def is_fit(self) -> bool:
        return self.P_ is not None

    def project(self, X: torch.Tensor) -> np.ndarray:
        assert self.P_ is not None and self.mean_ is not None, "FixedProjector2D not fitted"
        Xn = X.detach().cpu().numpy()
        return (Xn - self.mean_) @ self.P_


def tb_log_knn_convergence(
    writer: 'SummaryWriter', tag: str, step: int,
    projector: FixedProjector2D,
    train_feats: torch.Tensor,
    val_feats_prev0: Optional[torch.Tensor],  # baseline snapshot (first val) for trajectory; if None, will be set to current
    val_feats_now: torch.Tensor,
    tracked_idx: Optional[np.ndarray] = None,
    max_tracked: int = 200,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Log a 2D plot that shows how validation feature vectors move over time under a fixed projector.

    Returns potentially updated (val_feats_prev0, tracked_idx) for the caller to persist.
    """
    try:
        if not projector.is_fit():
            projector.fit(train_feats)
        # Choose tracked subset once for stable arrows
        if tracked_idx is None:
            rng = np.random.default_rng(123)
            n = val_feats_now.size(0)
            k = min(max_tracked, n)
            tracked_idx = rng.choice(n, size=k, replace=False)

        Ztr = projector.project(train_feats)
        Znow = projector.project(val_feats_now)
        if val_feats_prev0 is None:
            val_feats_prev0 = val_feats_now.detach().clone()
        Z0 = projector.project(val_feats_prev0)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5), dpi=150)
        # Background: train cloud for context
        ax.scatter(Ztr[:,0], Ztr[:,1], s=3, c='#cccccc', alpha=0.35, label='train bank')
        # Current validation
        ax.scatter(Znow[:,0], Znow[:,1], s=6, c='tab:blue', alpha=0.7, label='val now')
        # Trajectories (arrows) from first snapshot → now
        idx = tracked_idx
        ax.quiver(Z0[idx,0], Z0[idx,1], (Znow[idx,0]-Z0[idx,0]), (Znow[idx,1]-Z0[idx,1]),
                  angles='xy', scale_units='xy', scale=1.0, width=0.002, color='tab:orange', alpha=0.7)
        ax.set_title('kNN feature convergence (fixed PCA basis)')
        ax.set_xlim(projector.xlim_)
        ax.set_ylim(projector.ylim_)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(frameon=False, loc='lower right')
        import io, imageio
        buf = io.BytesIO()
        fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig)
        buf.seek(0)
        img = imageio.v2.imread(buf)
        chw = np.transpose(img, (2,0,1))
        writer.add_image(tag, chw, global_step=step)
        return val_feats_prev0, tracked_idx
    except Exception:
        return val_feats_prev0, tracked_idx
