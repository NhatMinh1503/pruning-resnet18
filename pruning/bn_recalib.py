"""BatchNorm re-calibration after pruning (no fine-tune).

Owner: Hoa.

Question
--------
At 90% sparsity Local pruning drops to 24.5% and Global to 57.1% before
fine-tuning, yet 10 epochs of fine-tuning bring both back to ~94-95%.
How much of that collapse is "real" (lost information) and how much is only
BatchNorm being out of date?

BatchNorm stores the mean/variance of every channel measured on the DENSE
model. After pruning, each conv produces activations with a different scale,
so BN normalizes with the wrong statistics and the error grows layer by layer.
Re-estimating those statistics needs only forward passes on a few thousand
training images: no labels, no gradients, no weight update, a few seconds
on CPU (idea used by e.g. EagleEye, Li et al. 2020 "adaptive BN").

What the script does
--------------------
For every (method, sparsity):
    prune baseline -> eval (acc_raw) -> reset BN stats -> forward n_calib
    TRAIN images -> eval (acc_bn_recalib)
The calibration images use the TRAINING augmentation (RandomCrop + Flip) by
default, because that is the distribution the original BN statistics came
from. With clean images (--calib-clean) even the dense model drops
95.10 -> 93.63%, so the statistics must match what the weights were trained on.
Methods: local / global (prune.py) and structured (structured.py).
Weights are never changed: sparsity stays exactly the same.

Output: results/bn_recalib.csv
    method, sparsity, real_sparsity, acc_raw, acc_bn_recalib, n_calib, seed
and with --sweep also results/bn_recalib_ncalib.csv (accuracy vs n_calib).

Usage
-----
    python pruning/bn_recalib.py --checkpoint {CK}/resnet18_baseline.pt --data-dir /content/data
    python pruning/bn_recalib.py --checkpoint ... --sweep        # + n_calib sweep
    python pruning/bn_recalib.py --checkpoint ... --calib-clean --csv results/bn_recalib_clean.csv
    python pruning/bn_recalib.py --random-init --smoke           # pipeline test, fake data

Data: uses torchvision CIFAR10 when the dataset is complete. If it is not
(e.g. a half-downloaded tar), it falls back to reading whatever
cifar-10-batches-py/data_batch_* and test_batch files can be unpickled.
"""
import argparse
import copy
import csv
import glob
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from pruning.prune import count_sparsity, global_prune, local_prune, make_permanent  # noqa: E402
from pruning.structured import channel_prune  # noqa: E402
from training.common import (CIFAR10_MEAN, CIFAR10_STD, SEED, build_model,  # noqa: E402
                             evaluate, get_device, load_weights, set_seed)

CSV_HEADER = ["method", "sparsity", "real_sparsity", "acc_raw", "acc_bn_recalib",
              "n_calib", "seed"]
SWEEP_HEADER = ["method", "sparsity", "n_calib", "acc_bn_recalib", "seed"]


# --------------------------------------------------------------------------- #
# BN re-calibration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def recalibrate_bn(model, loader, device):
    """Replace every BatchNorm's running mean/var with statistics measured on
    `loader` (cumulative average over all its images). Only BN buffers change."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # None = exact cumulative average, independent of batch order
    model.to(device).train()
    for x, _ in loader:
        model(x.to(device, non_blocking=True))
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    return model.eval()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class _RawCIFAR(torch.utils.data.Dataset):
    """CIFAR-10 read straight from the python pickles that can be opened."""

    def __init__(self, files, transform):
        xs, ys = [], []
        for f in files:
            with open(f, "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            xs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys += list(d[b"labels"])
        self.x, self.y, self.tf = np.concatenate(xs), ys, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.fromarray(self.x[i])), self.y[i]


def _readable(files):
    ok = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                pickle.load(fh, encoding="bytes")
            ok.append(f)
        except Exception:  # truncated / missing
            print(f"  (skip unreadable {os.path.basename(f)})")
    return ok


def get_datasets(data_dir, smoke=False, aug=False):
    """(train_set, test_set). The test set always uses the test transform; the
    calibration (train) set uses it too, unless aug=True (RandomCrop + Flip,
    i.e. the same images the BN statistics were collected on during training)."""
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]) if aug else tf
    if smoke:
        return (torchvision.datasets.FakeData(512, (3, 32, 32), 10, tf, random_offset=0),
                torchvision.datasets.FakeData(256, (3, 32, 32), 10, tf, random_offset=1000))
    root = os.path.join(data_dir, "cifar-10-batches-py")
    # download only when nothing is there yet (a broken local copy -> raw fallback)
    download = not os.path.isdir(root)
    try:
        return (torchvision.datasets.CIFAR10(data_dir, train=True, download=download, transform=train_tf),
                torchvision.datasets.CIFAR10(data_dir, train=False, download=download, transform=tf))
    except Exception as e:  # incomplete dataset and/or no network
        print(f"torchvision CIFAR10 unavailable ({type(e).__name__}); reading raw batches")
    train_files = _readable(sorted(glob.glob(os.path.join(root, "data_batch_*"))))
    test_files = _readable([os.path.join(root, "test_batch")])
    if not train_files or not test_files:
        sys.exit(f"No usable CIFAR-10 batches in {root}")
    print(f"  train batches: {[os.path.basename(f) for f in train_files]}")
    return _RawCIFAR(train_files, train_tf), _RawCIFAR(test_files, tf)


def calib_loader(train_set, n, seed=SEED, batch_size=128):
    """Fixed random subset of n training images (same subset for every model)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_set), generator=g)[:n].tolist()
    return torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, idx),
                                       batch_size=batch_size, shuffle=False, num_workers=0)


# --------------------------------------------------------------------------- #
def prune(base, method, s):
    if method == "structured":
        return channel_prune(base, s)
    m = {"local": local_prune, "global": global_prune}[method](base, s)
    make_permanent(m)
    return m


def real_sparsity(model, method, s):
    # structured: fraction of inner channels removed (weights are gone, not zero)
    return s if method == "structured" else count_sparsity(model)


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to resnet18_baseline.pt")
    src.add_argument("--random-init", action="store_true", help="untrained model (pipeline test)")
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--sparsities", default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--ratios", default="0.25,0.5,0.75", help="structured channel ratios")
    p.add_argument("--n-calib", type=int, default=2000)
    p.add_argument("--calib-clean", action="store_true",
                   help="calibrate on clean images (test transform) instead of the default "
                        "augmented ones; this lowers even the dense model by ~1.5 pts")
    p.add_argument("--sweep", action="store_true", help="also sweep n_calib at 0.9 / 0.5")
    p.add_argument("--csv", default=os.path.join(REPO_ROOT, "results", "bn_recalib.csv"))
    p.add_argument("--smoke", action="store_true", help="fake data")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    base = build_model()
    if args.checkpoint:
        load_weights(base, args.checkpoint)
    base.eval()

    train_set, test_set = get_datasets(args.data_dir, args.smoke, aug=not args.calib_clean)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    n_calib = min(args.n_calib, len(train_set))
    cal = calib_loader(train_set, n_calib)

    acc0 = evaluate(base.to(device), test_loader, device)
    # sanity check: re-calibrating the DENSE model should change almost nothing
    acc0_cal = evaluate(recalibrate_bn(copy.deepcopy(base), cal, device), test_loader, device)
    rows = [{"method": "baseline", "sparsity": 0.0, "real_sparsity": "0.0000",
             "acc_raw": f"{acc0:.2f}", "acc_bn_recalib": f"{acc0_cal:.2f}",
             "n_calib": n_calib, "seed": SEED}]
    print(f"baseline   raw={acc0:.2f}%  bn_recalib={acc0_cal:.2f}%", flush=True)

    jobs = [(m, round(float(s), 2)) for m in ("local", "global") for s in args.sparsities.split(",")]
    jobs += [("structured", round(float(r), 2)) for r in args.ratios.split(",")]
    for method, s in jobs:
        model = prune(base, method, s).to(device).eval()
        rs = real_sparsity(model, method, s)
        a_raw = evaluate(model, test_loader, device)
        a_cal = evaluate(recalibrate_bn(model, cal, device), test_loader, device)
        rows.append({"method": method, "sparsity": s, "real_sparsity": f"{rs:.4f}",
                     "acc_raw": f"{a_raw:.2f}", "acc_bn_recalib": f"{a_cal:.2f}",
                     "n_calib": n_calib, "seed": SEED})
        print(f"{method:10s} {s:.2f}  raw={a_raw:6.2f}%  bn_recalib={a_cal:6.2f}%  "
              f"(+{a_cal - a_raw:.2f})", flush=True)
    write_csv(args.csv, CSV_HEADER, rows)

    if args.sweep:
        srows = []
        for method, s in (("local", 0.9), ("global", 0.9), ("structured", 0.5)):
            model = prune(base, method, s).to(device)
            for n in (64, 256, 1000, 4000, 16000):
                n = min(n, len(train_set))
                a = evaluate(recalibrate_bn(model, calib_loader(train_set, n), device),
                             test_loader, device)
                srows.append({"method": method, "sparsity": s, "n_calib": n,
                              "acc_bn_recalib": f"{a:.2f}", "seed": SEED})
                print(f"  sweep {method} {s} n={n:5d}: {a:.2f}%", flush=True)
        write_csv(args.csv.replace(".csv", "_ncalib.csv"), SWEEP_HEADER, srows)


if __name__ == "__main__":
    main()
