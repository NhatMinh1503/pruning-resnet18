"""Figure: magnitude vs BN-folded ranking (one-shot / iterative), all + BN recalib.

Owner: Hoa.  Reads results/bn_aware.csv and results/bn_aware_stages.csv;
writes results/figures/q2_bn_aware.png.

    python pruning/plot_bn_aware.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"
SERIES = [("magnitude", "oneshot", "#2a78d6", "-", "|w|, one-shot"),
          ("bn_folded", "oneshot", "#eb6834", "--", "|w|·|γ|/σ, one-shot"),
          ("bn_folded", "iter5", "#eb6834", "-", "|w|·|γ|/σ, iterative (5 steps)")]
STAGES = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    d = pd.read_csv(os.path.join(R, "bn_aware.csv"))
    st = pd.read_csv(os.path.join(R, "bn_aware_stages.csv"))
    panels = [("global", "Global (unstructured)", "sparsity"),
              ("local", "Local (unstructured)", "sparsity"),
              ("structured", "Structured (channels)", "inner-channel ratio")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1, 0.8, 1.15]})
    for ax, (m, title, xlabel) in zip(axes, panels):
        dm = d[d.method == m]
        for crit, sched, c, ls, lab in SERIES:
            s = dm[(dm.criterion == crit) & (dm.schedule == sched)].sort_values("sparsity")
            if len(s):
                ax.plot(s.sparsity, s.acc_bn_recalib, color=c, ls=ls, lw=2, marker="o", ms=5,
                        markeredgecolor="white", markeredgewidth=1.5, label=lab)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        xs = sorted(dm.sparsity.unique())
        ax.set_xticks(xs)
        ax.set_xlim(xs[0] - 0.03, xs[-1] + 0.03)
        ax.set_ylim(0, 100)
        style(ax)
    axes[0].set_ylabel("test accuracy after BN recalib (%)", fontsize=9, color=MUTED)

    # where Global 0.9 puts its zeros
    ax = axes[3]
    g = st[(st.method == "global") & (st.sparsity == 0.9)]
    x = np.arange(len(STAGES))
    w = 0.27
    for i, (crit, sched, c, ls, lab) in enumerate(SERIES):
        s = g[(g.criterion == crit) & (g.schedule == sched)].set_index("stage").reindex(STAGES)
        ax.bar(x + (i - 1) * w, s.stage_sparsity * 100, w, color=c,
               alpha=0.45 if ls == "--" else 1.0, label=lab)
    ax.axhline(90, color=MUTED, lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of weights pruned (%)", fontsize=9, color=MUTED)
    ax.set_title("Global 0.9: where the zeros go", fontsize=10, color=INK, loc="left")
    style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Ranking by |w| vs by BN-folded weight |w|·|γ|/σ (no fine-tune, BN re-calibrated), "
                 "ResNet-18 / CIFAR-10", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = os.path.join(R, "figures", "q2_bn_aware.png")
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
