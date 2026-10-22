"""Which layers make Local pruning collapse at high sparsity? (Q2 follow-up)

Owner: Hoa.

Hypothesis: at 90% sparsity Local pruning (24.49% before fine-tune) is far
worse than Global (57.13%) because Local forces the tiny layers - fc, conv1
and the downsample convs - to lose 90% of their weights, while Global almost
leaves them alone.

Test: prune the baseline with Local while keeping some layers dense (or
pruning ONLY some layers) and measure test accuracy right after pruning,
with no fine-tuning. Excluding these layers barely changes the overall
sparsity (they hold < 2% of all weights), so any accuracy gain comes from
which layers are pruned, not from how much.

Output: results/ablation_layers.csv
    config, method, sparsity, excluded, real_sparsity, acc_raw

Usage (needs the CIFAR-10 test set; downloaded to --data-dir on first run):
    python pruning/ablate_layers.py --checkpoint path/to/resnet18_baseline.pt
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from pruning.prune import count_sparsity, global_prune, local_prune  # noqa: E402
from training.common import (SEED, build_model, evaluate, get_device,  # noqa: E402
                             get_loaders, load_weights, set_seed)

CSV_HEADER = ["config", "method", "sparsity", "excluded", "real_sparsity", "acc_raw"]
SMALL = {
    "fc": ["fc"],
    "conv1": ["conv1"],
    "downsample": ["layer2.0.downsample.0", "layer3.0.downsample.0", "layer4.0.downsample.0"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to resnet18_baseline.pt")
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--sparsities", default="0.8,0.9")
    p.add_argument("--csv", default=os.path.join(REPO_ROOT, "results", "ablation_layers.csv"))
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def pick_device() -> torch.device:
    """CUDA on Colab, Apple GPU (MPS) on a Mac, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return get_device()


def all_layer_names(model: nn.Module) -> list:
    return [name for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def configs(model: nn.Module) -> list:
    """(config name, method, excluded layer names) for every experiment."""
    names = all_layer_names(model)
    small = SMALL["fc"] + SMALL["conv1"] + SMALL["downsample"]
    only = lambda keep: [n for n in names if n not in keep]  # noqa: E731  exclude all but `keep`
    return [
        ("local: all layers", "local", []),
        ("local: keep fc dense", "local", SMALL["fc"]),
        ("local: keep fc + conv1 dense", "local", SMALL["fc"] + SMALL["conv1"]),
        ("local: keep fc + conv1 + downsample dense", "local", small),
        ("local: prune ONLY fc", "local", only(SMALL["fc"])),
        ("local: prune ONLY conv1", "local", only(SMALL["conv1"])),
        ("local: prune ONLY downsample", "local", only(SMALL["downsample"])),
        ("global: all layers", "global", []),
    ]


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = pick_device()
    sparsities = [round(float(s), 2) for s in args.sparsities.split(",")]

    _, test_loader = get_loaders(args.data_dir, num_workers=args.num_workers, seed=SEED)
    base = load_weights(build_model(), args.checkpoint)
    acc_base = evaluate(base.to(device), test_loader, device)
    print(f"device={device} baseline acc={acc_base:.2f}%\n")
    base.cpu()

    rows = [{"config": "baseline", "method": "none", "sparsity": 0.0, "excluded": "",
             "real_sparsity": 0.0, "acc_raw": f"{acc_base:.2f}"}]
    prune_fns = {"local": local_prune, "global": global_prune}
    for s in sparsities:
        for name, method, excluded in configs(base):
            model = prune_fns[method](base, s, exclude=excluded)
            real = count_sparsity(model)
            acc = evaluate(model.to(device), test_loader, device)
            # "prune ONLY x" excludes almost every layer; store the short form
            short = "all except " + name.split("ONLY ")[1] if "ONLY" in name else ";".join(excluded)
            rows.append({"config": name, "method": method, "sparsity": s, "excluded": short,
                         "real_sparsity": f"{real:.4f}", "acc_raw": f"{acc:.2f}"})
            print(f"s={s:.1f}  {name:42s} real={real:.4f}  acc={acc:6.2f}%", flush=True)
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
