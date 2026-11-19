"""Figure: how much accuracy does BN re-calibration alone bring back?

Owner: Hoa.  Reads results/bn_recalib.csv
and results/accuracy.csv; writes results/figures/q3_bn_recalib.png.

    python pruning/plot_bn_recalib.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"
RAW = "#8f8f8a"                                       # neutral: the starting point
BN, FT = "#2a78d6", "#eb6834"                         # categorical slots 1, 2


def main():
    bn = pd.read_csv(os.path.join(R, "bn_recalib.csv"))
    acc = pd.read_csv(os.path.join(R, "accuracy.csv"))
    base = float(bn.query("method == 'baseline'").acc_raw.iloc[0])

    panels = [("local", "Local (unstructured)", "sparsity"),
              ("global", "Global (unstructured)", "sparsity"),
              ("structured", "Structured (channels removed)", "inner-channel ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=150, sharey=True)
    for ax, (m, title, xlabel) in zip(axes, panels):
        d = bn[bn.method == m].sort_values("sparsity")
        series = [(d.sparsity, d.acc_raw, RAW, "raw (prune only)", "--"),
                  (d.sparsity, d.acc_bn_recalib, BN, "+ BN recalib", "-")]
        f = acc[acc.method == m].sort_values("sparsity")
        f = f[f.sparsity.isin(d.sparsity)]
        if len(f):
            series.append((f.sparsity, f.acc_after_finetune, FT, "+ fine-tune 10 ep", "-"))
        for x, y, c, lab, ls in series:
            ax.plot(x, y, color=c, lw=2, ls=ls, marker="o", ms=5, label=lab,
                    markeredgecolor="white", markeredgewidth=1.5)
        # end-of-line values, nudged apart so they never overlap
        ends = sorted(((float(y.iloc[-1]), float(x.iloc[-1])) for x, y, *_ in series), reverse=True)
        placed = []
        for v, x in ends:
            pos = min(v, placed[-1] - 5.5) if placed else v
            placed.append(pos)
            ax.annotate(f"{v:.1f}", (x, v), xytext=(x + 0.012, pos), va="center",
                        fontsize=8, color=INK)
        ax.axhline(base, color=MUTED, lw=1, ls=":")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        ax.set_xticks(list(d.sparsity))
        ax.set_xlim(d.sparsity.min() - 0.04, d.sparsity.max() + 0.08)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("test accuracy (%)", fontsize=9, color=MUTED)
    axes[0].set_ylim(0, 100)
    axes[0].text(axes[0].get_xlim()[0] + 0.01, base + 1.5, f"baseline {base:.2f}%",
                 fontsize=7, color=MUTED)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle("BN re-calibration only (no weight update) vs fine-tuning, ResNet-18 / CIFAR-10",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = os.path.join(R, "figures", "q3_bn_recalib.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
