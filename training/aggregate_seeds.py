"""Mean +- std over seeds for Q1-Q3.

    python training/aggregate_seeds.py

Reads   results/accuracy.csv (seed 42) + results/accuracy_seed*.csv (other seeds)
Writes  results/accuracy_seeds_summary.csv
        results/figures/q1_q3_seeds.png   (mean with +-1 std error bars)

Each seed has its own baseline (training/train.py --seed N), so the spread includes
the variance of baseline training, pruning and fine-tuning.
"""
import csv
import glob
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "accuracy_seeds_summary.csv")
FIG = os.path.join(RES, "figures", "q1_q3_seeds.png")

COLORS = {"local": "#2a78d6", "global": "#eb6834"}
LABELS = {"local": "Local (per layer)", "global": "Global (whole network)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_all():
    files = [os.path.join(RES, "accuracy.csv")] + sorted(glob.glob(os.path.join(RES, "accuracy_seed*.csv")))
    rows = []
    for f in files:
        if os.path.exists(f):
            rows += list(csv.DictReader(open(f, newline="")))
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["method"], round(float(r["sparsity"]), 2))
        groups.setdefault(key, []).append(r)
    out = []
    order = {"baseline": 0, "local": 1, "global": 2}
    for (m, s), rs in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])):
        b = [float(r["acc_before_finetune"]) for r in rs]
        a = [float(r["acc_after_finetune"]) for r in rs]
        sd = (lambda x: st.stdev(x) if len(x) > 1 else 0.0)
        out.append({"method": m, "sparsity": f"{s:.1f}", "n_seeds": len(rs),
                    "seeds": " ".join(sorted({r["seed"] for r in rs})),
                    "before_mean": f"{st.mean(b):.2f}", "before_std": f"{sd(b):.2f}",
                    "after_mean": f"{st.mean(a):.2f}", "after_std": f"{sd(a):.2f}"})
    return out


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = next((r for r in summary if r["method"] == "baseline"), None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title, ylim in ((axes[0], "before", "Q1/Q2  Right after pruning", (0, 100)),
                                 (axes[1], "after", "Q3  After 10-epoch fine-tune", (93, 96))):
        ax.set_facecolor(SURFACE)
        for m in ("local", "global"):
            rs = [r for r in summary if r["method"] == m]
            xs = [float(r["sparsity"]) for r in rs]
            ys = [float(r[f"{key}_mean"]) for r in rs]
            es = [float(r[f"{key}_std"]) for r in rs]
            ax.errorbar(xs, ys, yerr=es, color=COLORS[m], linewidth=2, marker="o", markersize=5,
                        capsize=4, label=LABELS[m])
        if base:
            ax.axhline(float(base[f"{key}_mean"]), color=INK2, linestyle="--", linewidth=1)
            ax.text(0.095, float(base[f"{key}_mean"]), f" baseline {base[f'{key}_mean']}%",
                    va="bottom", fontsize=9, color=INK2)
        n = max(int(r["n_seeds"]) for r in summary)
        ax.set_title(f"{title}  (mean ± std, n={n} seeds)", loc="left", fontsize=12,
                     fontweight="bold", color=INK)
        ax.set_xticks([0.1 * i for i in range(1, 10)])
        ax.set_xticklabels([f"{10 * i}%" for i in range(1, 10)])
        ax.set_ylim(*ylim)
        ax.set_xlabel("Sparsity", color=INK2)
        ax.set_ylabel("Test accuracy (%)", color=INK2)
        ax.grid(axis="y", color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, facecolor=SURFACE)
    print("saved", os.path.relpath(FIG, ROOT))


def main():
    summary = summarize(load_all())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"{len(summary)} rows -> {os.path.relpath(OUT, ROOT)}")
    for r in summary:
        print(f"  {r['method']:8s} {r['sparsity']}  n={r['n_seeds']}  "
              f"before {r['before_mean']}±{r['before_std']}  after {r['after_mean']}±{r['after_std']}")
    plot(summary)


if __name__ == "__main__":
    main()
