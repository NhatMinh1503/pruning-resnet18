"""Stage ablation figure (Q2): which stage breaks Local pruning?

Owner: Hoa.  Reads results/ablation_stages.csv; writes results/figures/q2_stage_ablation.png.

    python pruning/plot_ablation.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
LIGHT, DARK = "#86b6ef", "#1c5cab"  # 80% / 90% (ordinal ramp)
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"


def main():
    df = pd.read_csv(os.path.join(R, "ablation_stages.csv"))
    base = float(df.query("config == 'baseline'").acc_raw.iloc[0])
    stages = ["layer1", "layer2", "layer3", "layer4"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    w = 0.36
    for i, (s, color) in enumerate(((0.8, LIGHT), (0.9, DARK))):
        d = df[(df.sparsity == s) & df.config.str.startswith("local: prune ONLY")]
        acc = [float(d[d.config.str.endswith(st)].acc_raw.iloc[0]) for st in stages]
        share = [float(d[d.config.str.endswith(st)].real_sparsity.iloc[0]) / s for st in stages]
        xs = [k + (i - 0.5) * w for k in range(4)]
        ax.bar(xs, acc, width=w - 0.03, color=color, label=f"prune only this stage at {s:.0%}",
               edgecolor="white", linewidth=2)
        for x, a in zip(xs, acc):
            ax.text(x, a - 5, f"{a:.1f}", ha="center", fontsize=8, color="white" if color == DARK else INK)
    labels = [f"{st}\n({100 * sh:.1f}% of weights)" for st, sh in zip(stages, share)]
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.axhline(base, color=MUTED, lw=1, ls="--", label=f"baseline {base:.2f}%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test accuracy, no fine-tune (%)", fontsize=9, color=INK)
    ax.set_title("Local pruning breaks in layer2-3, not in layer4 (most weights)",
                 fontsize=10, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    out = os.path.join(R, "figures", "q2_stage_ablation.png")
    fig.savefig(out, facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
