"""Shared helpers for training/ (and benchmark/): model, data, train/eval loops.

Everyone who loads a checkpoint MUST build the model with build_model(),
otherwise load_state_dict() will fail (CIFAR-10 variant of ResNet-18).
"""
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

SEED = 42
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CSV_HEADER = ["method", "sparsity", "acc_before_finetune", "acc_after_finetune", "seed"]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted to 32x32 CIFAR images.

    Changes vs. ImageNet ResNet-18: conv1 is 3x3 stride 1 (not 7x7 stride 2),
    and the first maxpool is removed. ~11.17M parameters.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_weights(model: nn.Module, path: str, device="cpu") -> nn.Module:
    obj = torch.load(path, map_location=device)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state)
    return model


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def get_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 2,
                smoke: bool = False, seed: int = SEED):
    """Returns (train_loader, test_loader) for CIFAR-10.

    smoke=True uses tiny random FakeData (no download) to test the pipeline.
    """
    normalize = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), normalize])
    test_tf = T.Compose([T.ToTensor(), normalize])

    if smoke:
        train_set = torchvision.datasets.FakeData(256, (3, 32, 32), 10, train_tf, random_offset=0)
        test_set = torchvision.datasets.FakeData(128, (3, 32, 32), 10, test_tf, random_offset=1000)
    else:
        train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, worker_init_fn=_seed_worker, generator=g,
        persistent_workers=num_workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=num_workers > 0)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, scaler, device, after_step=None):
    """One epoch with mixed precision on GPU. after_step() runs after every
    optimizer step (used by fine-tuning to keep pruned weights at zero).
    Returns (mean_loss, train_acc_%)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if after_step is not None:
            after_step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Top-1 accuracy (%) on the given loader."""
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def make_scaler(device):
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


# --------------------------------------------------------------------------- #
# results/accuracy.csv
# --------------------------------------------------------------------------- #
def read_rows(csv_path: str):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def upsert_row(csv_path: str, row: dict) -> None:
    """Insert or replace the row with the same (method, sparsity)."""
    rows = [r for r in read_rows(csv_path)
            if not (r["method"] == str(row["method"]) and r["sparsity"] == str(row["sparsity"]))]
    rows.append({k: str(row[k]) for k in CSV_HEADER})
    order = {"baseline": 0, "local": 1, "global": 2}
    rows.sort(key=lambda r: (order.get(r["method"], 9), float(r["sparsity"])))
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
