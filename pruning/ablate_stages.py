"""Which STAGE breaks Local pruning at high sparsity? (Q2 follow-up to ablate_layers.py)

Owner: Hoa.

ablate_layers.py tests the tiny layers (fc, conv1, downsample). This script
adds two checks:

1. Stage ablation: prune ONLY layer1 / layer2 / layer3 / layer4 with Local
   pruning at the target sparsity, everything else dense. Global pruning
   barely touches layer2-3 but removes almost all of layer4
   (see results/layer_sparsity.csv), so if one stage is fragile it shows here.

2. Matched sparsity: "keep the small layers dense" lowers the real sparsity a
   little (0.9 -> ~0.886). The matched config raises the sparsity of the other
   layers so the whole network is exactly at the target again, which makes the
   comparison with "local: all layers" fair.

Accuracy is measured right after pruning (no fine-tune), like ablate_layers.py.

Output: results/ablation_stages.csv
    config, method, sparsity, layer_sparsity, real_sparsity, acc_raw

Usage:
    python pruning/ablate_stages.py --checkpoint path/to/resnet18_baseline.pt
"""
import argparse
import csv
import os
import sys

import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from pruning.ablate_layers import SMALL, all_layer_names, pick_device  # noqa: E402
from pruning.prune import count_sparsity, global_prune, local_prune  # noqa: E402
from training.common import (SEED, build_model, evaluate, get_loaders,  # noqa: E402
                             load_weights, set_seed)

CSV_HEADER = ["config", "method", "sparsity", "layer_sparsity", "real_sparsity", "acc_raw"]
STAGES = ["layer1", "layer2", "layer3", "layer4"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to resnet18_baseline.pt")
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--sparsities", default="0.8,0.9")
    p.add_argument("--csv", default=os.path.join(REPO_ROOT, "results", "ablation_stages.csv"))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="fake data (pipeline test only)")
    return p.parse_args()


def num_weights(model):
    return {n: m.weight.numel() for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def configs(model, s):
    """(config, method, per-layer sparsity, excluded layers) for target sparsity s."""
    names = all_layer_names(model)
    sizes = num_weights(model)
    small = SMALL["fc"] + SMALL["conv1"] + SMALL["downsample"]
    rest = sum(v for k, v in sizes.items() if k not in small)
    matched = s * sum(sizes.values()) / rest  # sparsity on the other layers so total == s
    out = [
        ("local: all layers", "local", s, []),
        ("local: keep small layers dense", "local", s, small),
        ("local: keep small layers dense (matched)", "local", round(matched, 4), small),
        ("global: all layers", "global", s, []),
    ]
    for st in STAGES:
        keep = [n for n in names if n.startswith(st + ".") and n not in small]
        out.append((f"local: prune ONLY {st}", "local", s, [n for n in names if n not in keep]))
    return out


def main():
    args = parse_args()
    set_seed(SEED)
    device = pick_device()
    sparsities = [round(float(x), 2) for x in args.sparsities.split(",")]
    _, test_loader = get_loaders(args.data_dir, num_workers=args.num_workers,
                                 smoke=args.smoke, seed=SEED)
    base = load_weights(build_model(), args.checkpoint)
    acc_base = evaluate(base.to(device), test_loader, device)
    base.cpu()
    print(f"device={device} baseline acc={acc_base:.2f}%\n")

    rows = [{"config": "baseline", "method": "none", "sparsity": 0.0, "layer_sparsity": 0.0,
             "real_sparsity": 0.0, "acc_raw": f"{acc_base:.2f}"}]
    fns = {"local": local_prune, "global": global_prune}
    for s in sparsities:
        for name, method, ls, excluded in configs(base, s):
            model = fns[method](base, ls, exclude=excluded)
            real = count_sparsity(model)
            acc = evaluate(model.to(device), test_loader, device)
            rows.append({"config": name, "method": method, "sparsity": s, "layer_sparsity": ls,
                         "real_sparsity": f"{real:.4f}", "acc_raw": f"{acc:.2f}"})
            print(f"s={s:.1f}  {name:42s} layer_s={ls:.4f} real={real:.4f} acc={acc:6.2f}%",
                  flush=True)
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
