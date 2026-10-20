"""Per-layer sparsity of Local vs Global pruning (research question Q2).

Owner: Hoa.

Prunes the baseline checkpoint with local_prune and global_prune at several
sparsities and records how many weights each layer loses. No GPU and no
dataset are needed: pruning only ranks and zeroes weights, so it runs on CPU
in a few seconds.

Outputs
-------
results/layer_sparsity.csv          method, sparsity, layer, num_params, layer_sparsity
results/figures/layer_sparsity.png  global per-layer sparsity + parameter count per layer

Usage
-----
    # real analysis (download resnet18_baseline.pt from the team Drive first)
    python pruning/analyze_layers.py --checkpoint path/to/resnet18_baseline.pt

    # pipeline test with an untrained model (numbers are NOT meaningful)
    python pruning/analyze_layers.py --random-init
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")  # write PNG files without a display
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from pruning.prune import count_sparsity, global_prune, layer_sparsity, local_prune  # noqa: E402
from training.common import SEED, build_model, load_weights  # noqa: E402

CSV_HEADER = ["method", "sparsity", "layer", "num_params", "layer_sparsity"]
PRUNE_FNS = {"local": local_prune, "global": global_prune}

# Ordinal one-hue ramp (light -> dark = low -> high sparsity), validated for a light surface.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK = "#2b2b2b"
INK_MUTED = "#6b6b6b"
GRID = "#e6e6e3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to resnet18_baseline.pt")
    src.add_argument("--random-init", action="store_true",
                     help="use an untrained model to test the pipeline")
    p.add_argument("--sparsities", default="0.3,0.5,0.7,0.9")
    p.add_argument("--csv", default=os.path.join(REPO_ROOT, "results", "layer_sparsity.csv"))
    p.add_argument("--fig", default=os.path.join(REPO_ROOT, "results", "figures",
                                                 "layer_sparsity.png"))
    return p.parse_args()


def load_model(args: argparse.Namespace) -> nn.Module:
    """CIFAR ResNet-18 with baseline weights, or random weights for a smoke test."""
    torch.manual_seed(SEED)
    model = build_model()
    if args.checkpoint:
        load_weights(model, args.checkpoint)
    return model.eval()


def layer_sizes(model: nn.Module) -> dict:
    """Number of weights in every Conv2d/Linear layer, keyed by layer name."""
    return {name: m.weight.numel()
            for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def collect_rows(model: nn.Module, sparsities: list) -> list:
    """Prune with every method x sparsity and return one row per layer."""
    sizes = layer_sizes(model)
    rows = []
    for method, prune_fn in PRUNE_FNS.items():
        for s in sparsities:
            pruned = prune_fn(model, s)
            real = count_sparsity(pruned)
            print(f"{method:6s} target={s:.1f} real={real:.4f}")
            for layer, ls in layer_sparsity(pruned):
                rows.append({"method": method, "sparsity": s, "layer": layer,
                             "num_params": sizes[layer], "layer_sparsity": round(ls, 4)})
    return rows


def write_csv(rows: list, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)


def short_name(layer: str) -> str:
    """'layer2.0.downsample.0' -> 'L2.0.ds', 'layer3.1.conv2' -> 'L3.1.c2'."""
    return (layer.replace("layer", "L").replace("downsample.0", "ds")
            .replace("conv", "c"))


def plot(rows: list, sparsities: list, path: str, title_note: str) -> None:
    """Top: global per-layer sparsity per target (local is flat at the target).
    Bottom: weights per layer, which explains where global pruning cuts most."""
    layers = [r["layer"] for r in rows if r["method"] == "global" and r["sparsity"] == sparsities[0]]
    sizes = [r["num_params"] for r in rows if r["method"] == "global" and r["sparsity"] == sparsities[0]]
    x = range(len(layers))

    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]})
    colors = RAMP[-len(sparsities):] if len(sparsities) <= len(RAMP) else None

    for i, s in enumerate(sparsities):
        color = colors[i] if colors else plt.cm.Blues(0.35 + 0.6 * i / max(1, len(sparsities) - 1))
        ys = [r["layer_sparsity"] for r in rows if r["method"] == "global" and r["sparsity"] == s]
        ax.axhline(s, color=color, lw=1, ls=(0, (4, 3)), alpha=0.7)  # local = target everywhere
        ax.plot(x, ys, color=color, lw=2, marker="o", ms=5,
                markeredgecolor="white", markeredgewidth=1.5, label=f"global {s:.0%}")

    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("layer sparsity", color=INK)
    ax.set_title(f"Per-layer sparsity: Global vs Local pruning{title_note}",
                 loc="left", color=INK, fontsize=12, pad=24)
    ax.text(0, 1.02, "solid = global (per layer)   dashed = local (every layer equals the target)",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
    ax.legend(loc="upper left", ncol=len(sparsities), frameon=False, fontsize=9, labelcolor=INK)

    ax_n.bar(x, sizes, color="#9ec5f4", width=0.7)
    ax_n.set_yscale("log")
    ax_n.set_ylabel("weights\n(log scale)", color=INK)
    ax_n.set_xticks(list(x))
    ax_n.set_xticklabels([short_name(l) for l in layers], rotation=60, ha="right", fontsize=8)

    for a in (ax, ax_n):
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(INK_MUTED)
        a.tick_params(colors=INK_MUTED)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sparsities = [round(float(s), 2) for s in args.sparsities.split(",")]
    model = load_model(args)

    rows = collect_rows(model, sparsities)
    write_csv(rows, args.csv)
    note = " (RANDOM INIT - test only)" if args.random_init else ""
    plot(rows, sparsities, args.fig, note)
    print(f"\nWrote {args.csv}\nWrote {args.fig}")


if __name__ == "__main__":
    main()
