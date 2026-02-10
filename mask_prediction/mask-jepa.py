import math  # базовая математика для косинуса и логарифма
import os  # пути и директории
import random  # рандом для сплитов и масок
import time  # timestamp run id
from typing import Any, Callable, Dict, List, Optional, Tuple  # простые типы для читаемости

import mlflow  # логирование экспериментов в sqlite через mlflow
import numpy as np  # numpy для ece/entropy метрик
import torch  # основной фреймворк
import torch.nn as nn  # слои нейросети
import torch.nn.functional as F  # функции типа normalize
import torchvision.transforms.functional as TF  # blur/jitter в robustness test
from datasets import load_dataset  # tiny imagenet из huggingface
from torch.utils.data import DataLoader, Dataset, Subset  # датасет и лоадеры
from torchvision import transforms  # базовые transform пайплайны


# ------------------
# Minimal configuration
# ------------------
RUN_ID = time.strftime("%Y%m%d-%H%M%S")  # id текущего запуска
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))  # одна строка выбора device

SEED = 42  # общий seed
NUM_CLASSES = 100  # tiny imagenet-100
IMAGE_SIZE = 64  # размер изображения на входе
PATCHES_PER_SIDE = 4  # как вы попросили: 4 патча по каждой оси
PATCH_SIZE = IMAGE_SIZE // PATCHES_PER_SIDE  # 64/4=16
NUM_PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE  # всего 16 токенов-патчей
MAX_VISIBLE_PATCHES = NUM_PATCHES // 2  # максимум видимых патчей 8

HIDDEN_DIM = 384  # размер скрытого пространства токенов
VIT_LAYERS = 10  # число transformer-блоков
VIT_HEADS = 8  # число attention-голов
FLOW_LAYERS = 5  # число coupling слоев в RealNVP
FLOW_HIDDEN = 768  # hidden размер MLP внутри coupling
FLOW_S_CLAMP = 2.0  # ограничение для log-scale

EPOCHS = 200  # число эпох
BATCH_SIZE = 600  # batch size
LR = 3e-4  # learning rate
WEIGHT_DECAY = 1e-4  # weight decay
GRAD_CLIP_MAX_NORM = 3.0  # clip градиента

TRAIN_RATIO = 0.9  # train/val split внутри официального train
TRAIN_NUM_WORKERS = 2  # воркеры train loader
VAL_NUM_WORKERS = 2  # воркеры val/test loader
PIN_MEMORY = (DEVICE.type == "cuda")  # pin memory только для cuda

MOMENTUM_INIT = 0.998  # старт EMA momentum teacher
MOMENTUM_FINAL = 1.0  # финальный EMA momentum teacher

LOG_EVERY_SAMPLES = 20_000  # частота train логов по samples_seen
VAL_EVERY_SAMPLES = 120_000  # частота val и checkpoint по samples_seen
VAL_MAX_BATCHES = 5  # ограничение батчей для быстрой periodic val
KNN_K = 20  # k в kNN

CHECKPOINT_DIR = os.path.abspath("./checkpoints")  # директория чекпоинтов
LAST_CKPT = os.path.join(CHECKPOINT_DIR, "mask_jepa_flow_last.pt")  # последний чекпоинт
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "mask_jepa_flow_best.pt")  # лучший по val nll

MLFLOW_DB_PATH = os.path.abspath("./mlruns/mlflow.db")  # sqlite файл
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"  # uri для mlflow
MLFLOW_EXPERIMENT = "mask-jepa-flow"  # имя эксперимента

TRAIN_MEAN_STD = None  # кэш mean/std train


# ------------------
# Data
# ------------------
class ToRGB:
    def __call__(self, image):
        return image.convert("RGB")  # приводим все входы к RGB


class TinyImageNetTorch(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset  # huggingface dataset
        self.transform = transform  # transform функция

    def __len__(self):
        return len(self.data)  # размер датасета

    def __getitem__(self, index):
        image = self.data[index]["image"]  # PIL image
        label = self.data[index]["label"]  # integer label
        if self.transform is not None:
            image = self.transform(image)  # transform в tensor [3,64,64]
        return image, label  # возвращаем (x,y)


def _filter_label_lt_num_classes(example):
    return int(example.get("label", -1)) < NUM_CLASSES  # оставляем только классы 0..99


def load_tinyimagenet_split(split: str):
    data = load_dataset("zh-plus/tiny-imagenet", split=split)  # загрузили split
    data = data.filter(_filter_label_lt_num_classes)  # фильтр классов
    return data  # возвращаем huggingface dataset


def compute_train_mean_std(train_raw) -> Tuple[np.ndarray, np.ndarray]:
    mean = torch.zeros(3)  # mean по каналам RGB
    std = torch.zeros(3)  # std по каналам RGB
    count = len(train_raw)  # число изображений
    for sample in train_raw:
        image = transforms.ToTensor()(sample["image"])  # [3,H,W] в [0,1]
        mean += image.mean(dim=(1, 2))  # среднее по пикселям каждого канала
        std += image.std(dim=(1, 2))  # std по пикселям каждого канала
    mean /= count  # финальный mean
    std /= count  # финальный std
    return mean.numpy(), std.numpy()  # numpy массивы для transforms.Normalize


def get_train_transform(mean: np.ndarray, std: np.ndarray):
    return transforms.Compose(  # отдельная функция train transform
        [
            ToRGB(),  # гарантируем RGB
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # только resize
            transforms.ToTensor(),  # tensor [3,64,64]
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),  # нормализация
        ]
    )


def get_val_transform(mean: np.ndarray, std: np.ndarray):
    return transforms.Compose(  # отдельная функция val/test transform
        [
            ToRGB(),  # гарантируем RGB
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # только resize
            transforms.ToTensor(),  # tensor [3,64,64]
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),  # нормализация
        ]
    )


def split_indices(total_size: int, seed: int, train_ratio: float) -> Tuple[List[int], List[int]]:
    indices = list(range(total_size))  # все индексы
    rng = random.Random(seed)  # локальный генератор для воспроизводимости
    rng.shuffle(indices)  # случайная перестановка
    train_size = int(train_ratio * total_size)  # сколько в train
    train_idx = indices[:train_size]  # train часть
    val_idx = indices[train_size:]  # val часть
    return train_idx, val_idx  # два списка индексов


def build_dataloaders() -> Dict[str, Any]:
    global TRAIN_MEAN_STD  # используем кэш для mean/std

    train_raw = load_tinyimagenet_split("train")  # официальный train
    test_raw = load_tinyimagenet_split("valid")  # официальный valid как hold-out test

    if TRAIN_MEAN_STD is None:
        TRAIN_MEAN_STD = compute_train_mean_std(train_raw)  # считаем статистики один раз
    mean, std = TRAIN_MEAN_STD  # распаковали

    train_transform = get_train_transform(mean, std)  # train transform
    val_transform = get_val_transform(mean, std)  # val/test transform

    train_dataset_all = TinyImageNetTorch(train_raw, transform=train_transform)  # train view
    val_dataset_all = TinyImageNetTorch(train_raw, transform=val_transform)  # eval view того же raw train
    test_dataset = TinyImageNetTorch(test_raw, transform=val_transform)  # hold-out test

    train_idx, val_idx = split_indices(len(train_dataset_all), SEED, TRAIN_RATIO)  # 90/10 split

    train_dataset = Subset(train_dataset_all, train_idx)  # train subset
    val_dataset = Subset(val_dataset_all, val_idx)  # val subset
    train_eval_dataset = Subset(val_dataset_all, train_idx)  # train subset с eval transform

    train_loader = DataLoader(  # train loader
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(  # val loader
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    train_eval_loader = DataLoader(  # train eval loader для kNN/probe
        train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(  # hold-out test loader
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return {
        "train": train_loader,  # train batches
        "val": val_loader,  # periodic validation batches
        "train_eval": train_eval_loader,  # train features for kNN/probe
        "test": test_loader,  # final test batches
        "train_size": len(train_dataset),  # число train примеров
        "val_size": len(val_dataset),  # число val примеров
        "test_size": len(test_dataset),  # число test примеров
    }


# ------------------
# Backbone
# ------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()  # инициализация nn.Module
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)  # self-attention [B,T,D]
        self.ln1 = nn.LayerNorm(hidden_dim)  # первый layer norm
        self.ffn = nn.Sequential(  # position-wise feed-forward
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)  # второй layer norm

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)  # attention выход [B,T,D]
        x = self.ln1(x + attn_out)  # residual + norm
        x = self.ln2(x + self.ffn(x))  # residual + norm
        return x  # новые токены [B,T,D]


class ViTBackbone(nn.Module):
    def __init__(self, image_size: int, patch_size: int, hidden_dim: int, num_layers: int, num_heads: int):
        super().__init__()  # инициализация nn.Module
        self.grid = image_size // patch_size  # патчей по стороне
        self.num_patches = self.grid * self.grid  # всего патч-токенов
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)  # [B,3,64,64] -> [B,D,4,4]
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))  # позиционные эмбеддинги [1,16,D]
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])  # стек трансформеров
        self.norm = nn.LayerNorm(hidden_dim)  # финальная нормализация

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)  # [B,D,4,4]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B,16,D]
        tokens = tokens + self.pos_embed  # добавили позицию
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=None)  # обычный full attention
        tokens = self.norm(tokens)  # финальная нормализация
        return tokens  # [B,16,D]


class Student(nn.Module):
    def __init__(self, backbone: ViTBackbone):
        super().__init__()  # инициализация nn.Module
        self.backbone = backbone  # ViT backbone

    def encode_context(self, x: torch.Tensor, visible_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)  # patch tokens без блоков [B,16,D]
        pos = self.backbone.pos_embed  # позиции [1,16,D]
        batch_size = x.shape[0]  # реальный размер батча

        context_tokens = torch.zeros((batch_size, MAX_VISIBLE_PATCHES, HIDDEN_DIM), device=x.device, dtype=tokens.dtype)  # padded контекст [B,8,D]
        context_pad = torch.ones((batch_size, MAX_VISIBLE_PATCHES), device=x.device, dtype=torch.bool)  # True=pad [B,8]

        for batch_id in range(batch_size):
            keep_idx = visible_mask[batch_id].nonzero(as_tuple=False).squeeze(1)  # индексы видимых патчей
            length = int(keep_idx.numel())  # сколько реально видно
            chosen_tokens = tokens[batch_id, keep_idx] + pos[0, keep_idx]  # контекст с позицией [L,D]
            context_tokens[batch_id, :length] = chosen_tokens  # кладем в начало padded буфера
            context_pad[batch_id, :length] = False  # эти позиции не паддинг

        for block in self.backbone.blocks:
            context_tokens = block(context_tokens, key_padding_mask=context_pad)  # attention только по видимым через mask
        context_tokens = self.backbone.norm(context_tokens)  # нормализация
        return context_tokens, context_pad  # [B,8,D], [B,8]


class Teacher(nn.Module):
    def __init__(self, backbone: ViTBackbone):
        super().__init__()  # инициализация nn.Module
        self.backbone = backbone  # ViT backbone
        for parameter in self.parameters():
            parameter.requires_grad = False  # teacher только EMA, без backprop

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # полный forward teacher [B,16,D]

    @torch.no_grad()
    def ema_update(self, student: Student, momentum: float):
        for teacher_param, student_param in zip(self.backbone.parameters(), student.backbone.parameters()):
            teacher_param.data.mul_(momentum)  # m * theta_teacher
            teacher_param.data.add_(student_param.data, alpha=(1.0 - momentum))  # + (1-m) * theta_student


# ------------------
# Minimal RealNVP
# ------------------
class CouplingLayer(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int, swap_halves: bool):
        super().__init__()  # инициализация nn.Module
        self.dim = dim  # размер входного токена D
        self.half = dim // 2  # половина размерности
        self.swap_halves = swap_halves  # чередование половинок между слоями
        self.net = nn.Sequential(  # простая MLP для scale/shift
            nn.Linear(self.half + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.half),
        )

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.swap_halves:
            x_keep = x[:, :self.half]  # первая половина — условие
            x_change = x[:, self.half:]  # вторая половина — трансформируем
        else:
            x_keep = x[:, self.half:]  # вторая половина — условие
            x_change = x[:, :self.half]  # первая половина — трансформируем
        return x_keep, x_change  # две половины

    def _merge(self, x_keep: torch.Tensor, x_change: torch.Tensor) -> torch.Tensor:
        if not self.swap_halves:
            y = torch.cat([x_keep, x_change], dim=1)  # обратно [keep | change]
        else:
            y = torch.cat([x_change, x_keep], dim=1)  # обратно [change | keep]
        return y  # итоговый токен [N,D]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_keep, x_change = self._split(x)  # split входа
        h = torch.cat([x_keep, cond], dim=1)  # условие для MLP
        s_t = self.net(h)  # [N,2*half]
        s = s_t[:, :self.half]  # scale часть
        t = s_t[:, self.half:]  # shift часть
        s = FLOW_S_CLAMP * torch.tanh(s)  # ограничиваем scale для стабильности
        y_change = x_change * torch.exp(s) + t  # affine transform
        y = self._merge(x_keep, y_change)  # склеиваем обратно
        log_det = s.sum(dim=1)  # log det Jacobian
        return y, log_det  # transformed token + log_det

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_keep, y_change = self._split(y)  # split выхода
        h = torch.cat([y_keep, cond], dim=1)  # условие для MLP
        s_t = self.net(h)  # [N,2*half]
        s = s_t[:, :self.half]  # scale часть
        t = s_t[:, self.half:]  # shift часть
        s = FLOW_S_CLAMP * torch.tanh(s)  # то же ограничение
        x_change = (y_change - t) * torch.exp(-s)  # обратное affine
        x = self._merge(y_keep, x_change)  # склейка обратно
        log_det = (-s).sum(dim=1)  # log det для inverse
        return x, log_det  # latent + log_det


class CondRealNVPFlow(nn.Module):
    def __init__(self, dim: int, cond_dim: int, num_layers: int, hidden_dim: int):
        super().__init__()  # инициализация nn.Module
        layers = []  # список coupling слоев
        for layer_id in range(num_layers):
            swap_halves = bool(layer_id % 2 == 1)  # чередуем half split
            layer = CouplingLayer(dim, cond_dim, hidden_dim, swap_halves)  # один слой
            layers.append(layer)  # добавили слой
        self.layers = nn.ModuleList(layers)  # регистрируем список слоев
        self.dim = dim  # размер D

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z  # текущий токен
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)  # накопление log_det
        for layer in self.layers:
            x, layer_log_det = layer(x, cond)  # один coupling шаг
            log_det = log_det + layer_log_det  # суммируем log_det
        return x, log_det  # x и полный log_det

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x  # текущий токен
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # накопление log_det
        for layer in reversed(self.layers):
            z, layer_log_det = layer.inverse(z, cond)  # один обратный шаг
            log_det = log_det + layer_log_det  # суммируем log_det
        return z, log_det  # latent z и полный log_det

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z, log_det = self.inverse(x, cond)  # переводим x в latent
        quad = z.pow(2).sum(dim=1)  # квадратичная форма для N(0,I)
        norm_const = self.dim * math.log(2.0 * math.pi)  # нормировочная константа
        log_base = -0.5 * (quad + norm_const)  # log p0(z)
        log_prob = log_base + log_det  # change of variables
        return log_prob  # log p(x|cond)


# ------------------
# Masking and loss
# ------------------
def sample_visible_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, NUM_PATCHES), dtype=torch.bool, device=device)  # [B,16]
    for batch_id in range(batch_size):
        visible_count = random.randint(1, MAX_VISIBLE_PATCHES)  # от 1 до 8
        visible_idx = torch.randperm(NUM_PATCHES, device=device)[:visible_count]  # индексы видимых патчей
        mask[batch_id, visible_idx] = True  # помечаем видимые
    return mask  # bool mask [B,16]


def apply_visible_mask(x: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
    patch_mask = visible_mask.view(x.shape[0], 1, PATCHES_PER_SIDE, PATCHES_PER_SIDE).float()  # [B,1,4,4]
    pixel_mask = patch_mask.repeat_interleave(PATCH_SIZE, dim=2).repeat_interleave(PATCH_SIZE, dim=3)  # [B,1,64,64]
    x_masked = x * pixel_mask  # нулим невидимые патчи
    return x_masked  # masked image [B,3,64,64]


def build_condition(student: Student, context_tokens: torch.Tensor, context_pad: torch.Tensor) -> torch.Tensor:
    visible = (~context_pad).float().unsqueeze(-1)  # [B,8,1], 1 где реальный токен
    pooled_sum = (context_tokens * visible).sum(dim=1)  # [B,D], сумма по контексту
    pooled_count = visible.sum(dim=1).clamp_min(1.0)  # [B,1], число видимых токенов
    pooled = pooled_sum / pooled_count  # [B,D], mean pooling
    pos = student.backbone.pos_embed.expand(context_tokens.shape[0], -1, -1)  # [B,16,D]
    cond = pooled.unsqueeze(1) + pos  # [B,16,D], контекст + позиция каждого целевого токена
    return cond  # condition tokens


def compute_loss(flow: CondRealNVPFlow, teacher_tokens: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
    token_flat = teacher_tokens.reshape(-1, HIDDEN_DIM)  # [B*16,D]
    cond_flat = condition_tokens.reshape(-1, HIDDEN_DIM)  # [B*16,D]
    log_prob = flow.log_prob(token_flat, cond_flat)  # [B*16]
    nll = -log_prob.mean()  # scalar nll
    return nll  # основной loss


def momentum_schedule(samples_seen: int, total_samples: int) -> float:
    if total_samples <= 1:
        return MOMENTUM_FINAL  # тривиальный случай короткого запуска
    progress = samples_seen / (total_samples - 1)  # прогресс от 0 до 1
    progress = min(1.0, max(0.0, progress))  # ограничили диапазон
    cos_value = math.cos(math.pi * progress)  # cos по прогрессу
    smooth_weight = 0.5 * (1.0 + cos_value)  # переводим в [0,1]
    momentum_delta = MOMENTUM_FINAL - MOMENTUM_INIT  # диапазон momentum
    momentum = MOMENTUM_FINAL - momentum_delta * smooth_weight  # итоговое значение
    return momentum  # текущий EMA momentum


@torch.no_grad()
def batch_nll(x_input: torch.Tensor, x_target: torch.Tensor, student: Student, teacher: Teacher, flow: CondRealNVPFlow) -> torch.Tensor:
    batch_size = x_input.shape[0]  # реальный размер батча
    visible_mask = sample_visible_mask(batch_size, x_input.device)  # [B,16]
    x_masked = apply_visible_mask(x_input, visible_mask)  # masked input
    teacher_tokens = teacher(x_target)  # teacher target tokens [B,16,D]
    context_tokens, context_pad = student.encode_context(x_masked, visible_mask)  # context [B,8,D], pad [B,8]
    condition_tokens = build_condition(student, context_tokens, context_pad)  # condition [B,16,D]
    nll = compute_loss(flow, teacher_tokens, condition_tokens)  # scalar nll
    return nll  # значение nll на батче


# ------------------
# Metrics
# ------------------
@torch.no_grad()
def extract_global_embeddings(backbone: ViTBackbone, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()  # eval mode
    features: List[torch.Tensor] = []  # список батчевых feature
    labels: List[torch.Tensor] = []  # список батчевых labels
    for x, y in loader:
        x = x.to(device, non_blocking=True)  # отправили батч на устройство
        tokens = backbone(x)  # [B,16,D]
        pooled = tokens.mean(dim=1)  # [B,D], глобальная эмбеддинговая фича
        features.append(pooled.cpu())  # складываем на cpu
        labels.append(y.cpu())  # складываем labels на cpu
    all_features = torch.cat(features, dim=0)  # [N,D]
    all_labels = torch.cat(labels, dim=0)  # [N]
    return all_features, all_labels  # итоговые тензоры


@torch.no_grad()
def knn_eval(train_features: torch.Tensor, train_labels: torch.Tensor, eval_features: torch.Tensor, eval_labels: torch.Tensor, k: int = KNN_K) -> float:
    train_features = F.normalize(train_features, dim=1)  # l2 normalize train features
    eval_features = F.normalize(eval_features, dim=1)  # l2 normalize eval features
    similarity = eval_features @ train_features.t()  # cosine similarity [N_eval,N_train]
    topk_idx = similarity.topk(k, dim=1).indices  # индексы k соседей
    topk_labels = train_labels[topk_idx]  # labels соседей
    pred = torch.mode(topk_labels, dim=1).values  # majority vote
    acc = (pred == eval_labels).float().mean().item()  # accuracy
    return float(acc)  # scalar accuracy


def compute_ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)  # границы confidence bins
    total = len(confidence)  # число примеров
    ece = 0.0  # аккумулятор ece
    for bin_id in range(bins):
        left = edges[bin_id]  # левая граница bin
        right = edges[bin_id + 1]  # правая граница bin
        if bin_id < bins - 1:
            mask = (confidence >= left) & (confidence < right)  # обычный полузакрытый bin
        else:
            mask = (confidence >= left) & (confidence <= right)  # последний bin включает 1.0
        if not np.any(mask):
            continue  # пустой bin пропускаем
        bin_conf = confidence[mask].mean()  # средняя confidence в bin
        bin_acc = correct[mask].mean()  # средняя accuracy в bin
        bin_weight = mask.sum() / max(1, total)  # вес bin
        ece += abs(bin_acc - bin_conf) * bin_weight  # вклад bin в ECE
    return float(ece)  # финальный ECE


def compute_entropy_error(prob: np.ndarray, labels: np.ndarray, bins: int = 10) -> Dict[str, float]:
    eps = 1e-12  # маленькая константа для логарифма
    entropy = -np.sum(prob * np.log(prob + eps), axis=1)  # энтропия предсказаний [N]
    pred = np.argmax(prob, axis=1)  # предсказанный класс
    error = (pred != labels).astype(np.float32)  # 1 если ошибка, 0 если верно
    corr = np.corrcoef(entropy, error)[0, 1]  # корреляция entropy-error
    if np.isnan(corr):
        corr = 0.0  # fallback если мало точек/константа

    max_entropy = math.log(prob.shape[1] + eps)  # максимум энтропии для C классов
    edges = np.linspace(0.0, max_entropy, bins + 1)  # bins по энтропии
    mean_x: List[float] = []  # средняя энтропия по bin
    mean_y: List[float] = []  # средняя ошибка по bin

    for bin_id in range(bins):
        left = edges[bin_id]  # левая граница
        right = edges[bin_id + 1]  # правая граница
        if bin_id < bins - 1:
            mask = (entropy >= left) & (entropy < right)  # обычный bin
        else:
            mask = (entropy >= left) & (entropy <= right)  # последний bin
        if not np.any(mask):
            continue  # пустой bin пропускаем
        mean_x.append(float(entropy[mask].mean()))  # средняя энтропия bin
        mean_y.append(float(error[mask].mean()))  # средняя ошибка bin

    slope = 0.0  # наклон тренда error vs entropy
    if len(mean_x) >= 2:
        x = np.array(mean_x, dtype=np.float32)  # x точки
        y = np.array(mean_y, dtype=np.float32)  # y точки
        x_c = x - x.mean()  # center x
        y_c = y - y.mean()  # center y
        denom = float((x_c ** 2).sum())  # знаменатель линейного slope
        if denom > 0.0:
            slope = float((x_c * y_c).sum() / denom)  # slope

    return {
        "entropy_error_corr": float(corr),  # корреляция
        "entropy_error_slope": float(slope),  # наклон
    }


# ------------------
# Logging and checkpoints
# ------------------
def setup_mlflow():
    os.makedirs(os.path.dirname(MLFLOW_DB_PATH), exist_ok=True)  # директория sqlite
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # подключили sqlite backend
    mlflow.set_experiment(MLFLOW_EXPERIMENT)  # выбрали эксперимент


def log_metrics(phase: str, step: int, metrics: Dict[str, float]):
    payload: Dict[str, float] = {}  # словарь для mlflow
    for key, value in metrics.items():
        payload[f"{phase}/{key}"] = float(value)  # префикс phase
    mlflow.log_metrics(payload, step=int(step))  # лог с step=samples_seen


def save_checkpoint(student: Student, teacher: Teacher, flow: CondRealNVPFlow, optimizer: torch.optim.Optimizer, state: Dict[str, Any], val_nll: float):
    checkpoint = {
        "student": student.state_dict(),  # веса student
        "teacher": teacher.state_dict(),  # веса teacher
        "flow": flow.state_dict(),  # веса flow
        "optimizer": optimizer.state_dict(),  # состояние optimizer
        "samples_seen": state["samples_seen"],  # сколько сэмплов уже обработано
        "step": state["step"],  # шаги по батчам
        "epoch": state["epoch"],  # текущая эпоха
        "next_val_samples": state["next_val_samples"],  # следующий порог val
        "next_log_samples": state["next_log_samples"],  # следующий порог log
        "best_val_nll": state["best_val_nll"],  # лучший val nll
        "rng": {
            "py": random.getstate(),  # состояние random python
            "np": np.random.get_state(),  # состояние random numpy
            "torch": torch.get_rng_state(),  # состояние random torch
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,  # состояние cuda random если есть
        },
    }
    torch.save(checkpoint, LAST_CKPT)  # всегда сохраняем последний
    if val_nll < state["best_val_nll"]:
        state["best_val_nll"] = val_nll  # обновили best в state
        checkpoint["best_val_nll"] = state["best_val_nll"]  # синхронизировали значение
        torch.save(checkpoint, BEST_CKPT)  # сохранили лучший


def load_checkpoint_if_exists(student: Student, teacher: Teacher, flow: CondRealNVPFlow, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    state = {
        "samples_seen": 0,  # стартовое значение
        "step": 0,  # стартовый step
        "epoch": 0,  # стартовая эпоха
        "next_val_samples": 0,  # первая val сразу
        "next_log_samples": LOG_EVERY_SAMPLES,  # первый train log после порога
        "best_val_nll": float("inf"),  # best nll пока бесконечность
    }
    if not os.path.isfile(LAST_CKPT):
        return state  # если файла нет, возвращаем стартовый state

    checkpoint = torch.load(LAST_CKPT, map_location=DEVICE)  # загрузили чекпоинт
    student.load_state_dict(checkpoint["student"], strict=True)  # восстановили student
    teacher.load_state_dict(checkpoint["teacher"], strict=True)  # восстановили teacher
    flow.load_state_dict(checkpoint["flow"], strict=True)  # восстановили flow
    optimizer.load_state_dict(checkpoint["optimizer"])  # восстановили optimizer

    state["samples_seen"] = int(checkpoint.get("samples_seen", 0))  # восстановили samples_seen
    state["step"] = int(checkpoint.get("step", 0))  # восстановили step
    state["epoch"] = int(checkpoint.get("epoch", 0))  # восстановили epoch
    state["next_val_samples"] = int(checkpoint.get("next_val_samples", state["samples_seen"] + VAL_EVERY_SAMPLES))  # восстановили порог val
    state["next_log_samples"] = int(checkpoint.get("next_log_samples", state["samples_seen"] + LOG_EVERY_SAMPLES))  # восстановили порог логов
    state["best_val_nll"] = float(checkpoint.get("best_val_nll", float("inf")))  # восстановили best nll

    if "rng" in checkpoint:
        random.setstate(checkpoint["rng"]["py"])  # восстановили python random
        np.random.set_state(checkpoint["rng"]["np"])  # восстановили numpy random
        torch.set_rng_state(checkpoint["rng"]["torch"])  # восстановили torch random
        if torch.cuda.is_available() and checkpoint["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])  # восстановили cuda random

    print(f"Resumed from {LAST_CKPT} at samples_seen={state['samples_seen']}")  # инфо в консоль
    return state  # возвращаем восстановленный state


# ------------------
# Train / Val / Test
# ------------------
def train_step(batch: Tuple[torch.Tensor, torch.Tensor], student: Student, teacher: Teacher, flow: CondRealNVPFlow, optimizer: torch.optim.Optimizer, samples_seen: int, total_samples: int) -> Dict[str, float]:
    x, _ = batch  # берем только изображения
    x = x.to(DEVICE, non_blocking=True)  # переносим на device
    batch_size = x.shape[0]  # фактический размер батча

    visible_mask = sample_visible_mask(batch_size, x.device)  # [B,16], какие патчи видим
    x_masked = apply_visible_mask(x, visible_mask)  # masked input

    with torch.no_grad():
        teacher_tokens = teacher(x)  # target tokens [B,16,D]

    context_tokens, context_pad = student.encode_context(x_masked, visible_mask)  # context [B,8,D] и паддинг маска [B,8]
    condition_tokens = build_condition(student, context_tokens, context_pad)  # cond [B,16,D]

    nll = compute_loss(flow, teacher_tokens, condition_tokens)  # основной loss
    loss = nll  # в этой версии loss=nll

    optimizer.zero_grad(set_to_none=True)  # чистим градиенты
    loss.backward()  # backprop
    grad_norm = torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(flow.parameters()), GRAD_CLIP_MAX_NORM)  # клип градиента
    optimizer.step()  # шаг оптимизатора

    momentum = momentum_schedule(samples_seen, total_samples)  # текущий EMA momentum
    teacher.ema_update(student, momentum)  # обновили teacher от student

    return {
        "loss": float(loss.item()),  # train loss
        "nll": float(nll.item()),  # train nll
        "grad_norm": float(grad_norm),  # grad norm
        "visible_patches_mean": float(visible_mask.float().sum(dim=1).mean().item()),  # среднее число видимых патчей
        "batch_size": float(batch_size),  # размер батча
    }


@torch.no_grad()
def val(student: Student, teacher: Teacher, flow: CondRealNVPFlow, optimizer: torch.optim.Optimizer, loaders: Dict[str, DataLoader], state: Dict[str, Any]) -> Dict[str, float]:
    student.eval()  # eval mode student
    teacher.eval()  # eval mode teacher
    flow.eval()  # eval mode flow

    nll_sum = 0.0  # аккумулятор nll
    count = 0  # число батчей
    for batch_id, (x, _) in enumerate(loaders["val"]):
        if batch_id >= VAL_MAX_BATCHES:
            break  # ограничиваем periodic val по батчам
        x = x.to(DEVICE, non_blocking=True)  # на device
        nll = batch_nll(x, x, student, teacher, flow)  # nll на батче
        nll_sum += float(nll.item())  # добавили в сумму
        count += 1  # увеличили счетчик
    val_nll = nll_sum / max(1, count)  # средний val nll

    train_features, train_labels = extract_global_embeddings(teacher.backbone, loaders["train_eval"], DEVICE)  # фичи train split
    val_features, val_labels = extract_global_embeddings(teacher.backbone, loaders["val"], DEVICE)  # фичи val split
    val_knn = knn_eval(train_features, train_labels, val_features, val_labels, k=KNN_K)  # periodic kNN

    metrics = {"nll": val_nll, "knn": val_knn}  # словарь val метрик
    log_metrics("val", state["samples_seen"], metrics)  # лог val метрик
    save_checkpoint(student, teacher, flow, optimizer, state, val_nll)  # checkpoint внутри val

    student.train()  # возвращаем train mode
    flow.train()  # возвращаем train mode
    return metrics  # val метрики


@torch.no_grad()
def evaluate_nll(loader: DataLoader, student: Student, teacher: Teacher, flow: CondRealNVPFlow, corruption_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> float:
    student.eval()  # eval mode
    teacher.eval()  # eval mode
    flow.eval()  # eval mode

    nll_sum = 0.0  # сумма nll
    count = 0  # число батчей
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)  # на device
        if corruption_fn is None:
            x_input = x  # clean input
        else:
            x_input = corruption_fn(x)  # corrupted input
        nll = batch_nll(x_input, x, student, teacher, flow)  # nll на этом батче
        nll_sum += float(nll.item())  # аккумуляция
        count += 1  # счетчик
    return nll_sum / max(1, count)  # средний nll


def corruption_noise(x: torch.Tensor) -> torch.Tensor:
    return x + 0.1 * torch.randn_like(x)  # gaussian noise аугментация


def corruption_blur(x: torch.Tensor) -> torch.Tensor:
    return TF.gaussian_blur(x, kernel_size=[3, 3], sigma=[0.6, 0.6])  # blur аугментация


def corruption_jitter(x: torch.Tensor) -> torch.Tensor:
    y = x * 1.1  # изменяем яркость
    channel_mean = y.mean(dim=1, keepdim=True)  # среднее по каналам
    y = (y - channel_mean) * 1.1 + channel_mean  # слегка меняем насыщенность
    return y  # color jitter-like аугментация


def run_linear_probe(backbone: ViTBackbone, train_eval_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression  # простой линейный классификатор

    train_features, train_labels = extract_global_embeddings(backbone, train_eval_loader, DEVICE)  # признаки train
    val_features, val_labels = extract_global_embeddings(backbone, val_loader, DEVICE)  # признаки val
    test_features, test_labels = extract_global_embeddings(backbone, test_loader, DEVICE)  # признаки test

    classifier = LogisticRegression(max_iter=300, multi_class="multinomial", solver="lbfgs", n_jobs=-1)  # multinomial linear probe
    classifier.fit(train_features.numpy(), train_labels.numpy())  # обучение probe на frozen backbone features

    val_prob = classifier.predict_proba(val_features.numpy())  # вероятности на val
    test_prob = classifier.predict_proba(test_features.numpy())  # вероятности на test

    val_pred = np.argmax(val_prob, axis=1)  # предсказанные классы val
    test_pred = np.argmax(test_prob, axis=1)  # предсказанные классы test

    val_acc = float((val_pred == val_labels.numpy()).mean())  # val accuracy
    test_acc = float((test_pred == test_labels.numpy()).mean())  # test accuracy

    eps = 1e-12  # константа для логарифма
    val_nll = float(-np.log(val_prob[np.arange(len(val_labels)), val_labels.numpy()] + eps).mean())  # val class nll
    test_nll = float(-np.log(test_prob[np.arange(len(test_labels)), test_labels.numpy()] + eps).mean())  # test class nll

    test_conf = np.max(test_prob, axis=1)  # confidence=max prob
    test_correct = (test_pred == test_labels.numpy()).astype(np.float32)  # correctness 0/1
    test_ece = compute_ece(test_conf, test_correct, bins=15)  # ECE по confidence bins
    entropy_metrics = compute_entropy_error(test_prob, test_labels.numpy(), bins=10)  # entropy-error анализ

    return {
        "probe_val_acc": val_acc,  # val accuracy probe
        "probe_val_nll": val_nll,  # val nll probe
        "probe_test_acc": test_acc,  # test accuracy probe
        "probe_test_nll": test_nll,  # test nll probe
        "probe_test_ece": test_ece,  # test ECE probe
        "probe_entropy_error_corr": entropy_metrics["entropy_error_corr"],  # corr entropy-error
        "probe_entropy_error_slope": entropy_metrics["entropy_error_slope"],  # slope entropy-error
    }


@torch.no_grad()
def test(student: Student, teacher: Teacher, flow: CondRealNVPFlow, loaders: Dict[str, DataLoader], state: Dict[str, Any]) -> Dict[str, float]:
    clean_nll = evaluate_nll(loaders["test"], student, teacher, flow, corruption_fn=None)  # clean nll
    noise_nll = evaluate_nll(loaders["test"], student, teacher, flow, corruption_fn=corruption_noise)  # noise robustness nll
    blur_nll = evaluate_nll(loaders["test"], student, teacher, flow, corruption_fn=corruption_blur)  # blur robustness nll
    jitter_nll = evaluate_nll(loaders["test"], student, teacher, flow, corruption_fn=corruption_jitter)  # jitter robustness nll

    train_features, train_labels = extract_global_embeddings(teacher.backbone, loaders["train_eval"], DEVICE)  # train features for kNN
    test_features, test_labels = extract_global_embeddings(teacher.backbone, loaders["test"], DEVICE)  # test features for kNN
    test_knn = knn_eval(train_features, train_labels, test_features, test_labels, k=KNN_K)  # test kNN

    probe_metrics = run_linear_probe(teacher.backbone, loaders["train_eval"], loaders["val"], loaders["test"])  # final linear probe

    metrics = {
        "nll": clean_nll,  # clean test nll
        "knn": test_knn,  # clean test knn
        "noise_nll": noise_nll,  # robustness nll noise
        "blur_nll": blur_nll,  # robustness nll blur
        "jitter_nll": jitter_nll,  # robustness nll jitter
    }
    metrics.update(probe_metrics)  # добавляем probe/cali метрики
    log_metrics("test", state["samples_seen"], metrics)  # логируем финальные test метрики
    return metrics  # возвращаем финальные метрики


def train(student: Student, teacher: Teacher, flow: CondRealNVPFlow, optimizer: torch.optim.Optimizer, loaders: Dict[str, DataLoader], state: Dict[str, Any]):
    total_samples = EPOCHS * loaders["train_size"]  # сколько сэмплов будет за весь run

    if state["samples_seen"] == 0:
        print("[val] initial validation at samples_seen=0")  # стартовая val на необученной модели
        val_metrics = val(student, teacher, flow, optimizer, loaders, state)  # первая val
        print(f"[val] nll={val_metrics['nll']:.4f} knn={val_metrics['knn']:.4f}")  # печать первой val
        state["next_val_samples"] = VAL_EVERY_SAMPLES  # следующий порог val

    for epoch in range(state["epoch"], EPOCHS):
        state["epoch"] = epoch  # фиксируем текущую эпоху в state
        student.train()  # train mode
        flow.train()  # train mode

        for batch in loaders["train"]:
            metrics = train_step(batch, student, teacher, flow, optimizer, state["samples_seen"], total_samples)  # один train step
            batch_size = int(metrics["batch_size"])  # размер текущего батча
            state["samples_seen"] += batch_size  # главный счетчик по samples
            state["step"] += 1  # счетчик батч-шагов

            while state["samples_seen"] >= state["next_log_samples"]:
                log_metrics("train", state["samples_seen"], metrics)  # train лог с step=samples_seen
                state["next_log_samples"] += LOG_EVERY_SAMPLES  # следующий порог train логов

            while state["samples_seen"] >= state["next_val_samples"]:
                print(f"[val] samples_seen={state['samples_seen']} threshold={state['next_val_samples']}")  # печать триггера val
                val_metrics = val(student, teacher, flow, optimizer, loaders, state)  # val + checkpoint
                print(f"[val] nll={val_metrics['nll']:.4f} knn={val_metrics['knn']:.4f}")  # печать val метрик
                state["next_val_samples"] += VAL_EVERY_SAMPLES  # следующий порог val


# ------------------
# Entry point
# ------------------
def main():
    random.seed(SEED)  # seed python random
    np.random.seed(SEED)  # seed numpy random
    torch.manual_seed(SEED)  # seed torch random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)  # seed cuda random

    setup_mlflow()  # инициализация mlflow
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # директория под чекпоинты

    print("\n=== mask-jepa + minimal RealNVP ===")  # заголовок запуска
    print(f"device={DEVICE}")  # выбранный device
    print(f"mlflow_uri={MLFLOW_TRACKING_URI}")  # uri для mlflow
    print(f"experiment={MLFLOW_EXPERIMENT}")  # имя эксперимента
    print(f"checkpoint_dir={CHECKPOINT_DIR}")  # директория чекпоинтов

    loaders = build_dataloaders()  # строим loaders
    print(f"dataset sizes: train={loaders['train_size']} val={loaders['val_size']} test={loaders['test_size']}")  # размеры сплитов
    print(f"batch_size={BATCH_SIZE} epochs={EPOCHS}")  # базовые training параметры

    student = Student(ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)).to(DEVICE)  # student model
    teacher = Teacher(ViTBackbone(IMAGE_SIZE, PATCH_SIZE, HIDDEN_DIM, VIT_LAYERS, VIT_HEADS)).to(DEVICE)  # teacher model
    flow = CondRealNVPFlow(HIDDEN_DIM, HIDDEN_DIM, FLOW_LAYERS, FLOW_HIDDEN).to(DEVICE)  # conditional flow head

    teacher.ema_update(student, momentum=0.0)  # teacher := student на старте

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(flow.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)  # optimizer
    state = load_checkpoint_if_exists(student, teacher, flow, optimizer)  # resume если есть ckpt

    params = {
        "seed": SEED,
        "device": str(DEVICE),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "image_size": IMAGE_SIZE,
        "patches_per_side": PATCHES_PER_SIDE,
        "patch_size": PATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "vit_layers": VIT_LAYERS,
        "vit_heads": VIT_HEADS,
        "flow_layers": FLOW_LAYERS,
        "flow_hidden": FLOW_HIDDEN,
        "val_every_samples": VAL_EVERY_SAMPLES,
        "log_every_samples": LOG_EVERY_SAMPLES,
        "max_visible_patches": MAX_VISIBLE_PATCHES,
        "knn_k": KNN_K,
    }

    with mlflow.start_run(run_name=RUN_ID):
        mlflow.log_params(params)  # лог конфигурации запуска
        train(student, teacher, flow, optimizer, loaders, state)  # train + periodic val
        test_metrics = test(student, teacher, flow, loaders, state)  # final test
        print(f"[test] nll={test_metrics['nll']:.4f} knn={test_metrics['knn']:.4f}")  # финальная печать


if __name__ == "__main__":
    main()  # стандартная точка входа
