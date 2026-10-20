"""Plot results/accuracy.csv for Q1-Q3 -> training/figures/*.png

    python training/plot_results.py

Figures (English labels so they render on any machine without CJK fonts):
  q1_q2_accuracy_before_finetune.png  Q1 + Q2: accuracy right after pruning, local vs global
  q3_accuracy_after_finetune.png      Q3: accuracy after fine-tuning (zoomed axis), local vs global
  q3_recovery.png                     Q3: points recovered by fine-tuning (after - before)
  q4_flops_vs_latency.png             Q4: sparse FLOPs vs measured latency, % of baseline
                                      (needs results/efficiency.csv from benchmark/)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "results", "accuracy.csv")
EFF_PATH = os.path.join(REPO_ROOT, "results", "efficiency.csv")
OUT_DIR = os.path.join(REPO_ROOT, "training", "figures")

# Reference palette: categorical slots 1-2 (blue, orange) + neutral inks
COLORS = {"local": "#2a78d6", "global": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"local": "Local (per layer)", "global": "Global (whole network)"}


def load(path=CSV_PATH):
    data = {"local": [], "global": []}
    baseline = None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == "baseline":
                baseline = float(r["acc_after_finetune"])
            elif r["method"] in data:
                data[r["method"]].append((float(r["sparsity"]),
                                          float(r["acc_before_finetune"]),
                                          float(r["acc_after_finetune"])))
    for m in data:
        data[m].sort()
    return baseline, data


def style(ax, title, ylabel):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel("Sparsity (fraction of Conv/Linear weights set to 0)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def baseline_line(ax, baseline):
    ax.axhline(baseline, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.095, baseline, f" baseline {baseline:.2f}%", va="bottom", ha="left",
            fontsize=9, color=INK2)


def pct_ticks(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x * 100)}%" for x in xs])


def plot_before(baseline, data):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for m, rows in data.items():
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, color=COLORS[m], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[m])
        ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=INK)
    baseline_line(ax, baseline)
    pct_ticks(ax, [r[0] for r in data["local"]])
    ax.set_ylim(0, 100)
    ax.set_xlim(0.07, 0.97)
    style(ax, "Q1/Q2  Accuracy right after pruning (no fine-tune)", "Test accuracy (%)")
    ax.legend(frameon=False, loc="lower left")
    return fig


def plot_after(baseline, data):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for m, rows in data.items():
        xs = [r[0] for r in rows]
        ys = [r[2] for r in rows]
        ax.plot(xs, ys, color=COLORS[m], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[m])
        ax.annotate(f"{ys[-1]:.2f}%", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=INK)
    baseline_line(ax, baseline)
    pct_ticks(ax, [r[0] for r in data["local"]])
    ax.set_ylim(93.5, 95.5)
    ax.set_xlim(0.07, 0.97)
    style(ax, "Q3  Accuracy after 10-epoch fine-tune (y-axis zoomed: 93.5-95.5%)",
          "Test accuracy (%)")
    ax.legend(frameon=False, loc="lower left")
    return fig


def plot_recovery(baseline, data):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    xs = [r[0] for r in data["local"]]
    width = 0.035
    for i, (m, rows) in enumerate(data.items()):
        offs = (-0.5 + i) * (width + 0.004)
        rec = [r[2] - r[1] for r in rows]
        bars = ax.bar([x + offs for x in xs], rec, width=width, color=COLORS[m],
                      label=LABELS[m], edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, rec):
            if abs(v) >= 5:  # label only the large bars
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"+{v:.1f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, linewidth=1)
    pct_ticks(ax, xs)
    style(ax, "Q3  Points recovered by fine-tuning (after - before)",
          "Accuracy change (percentage points)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.02, 0.72, "10-30%: tiny negative bars (-0.2 to -0.4)\n= fine-tune slightly lowered accuracy",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
    return fig


def plot_q4(path=EFF_PATH):
    """Finetuned checkpoints only, everything as % of the baseline row."""
    rows = list(csv.DictReader(open(path, newline="")))
    base = next(r for r in rows if r["method"] == "baseline")
    b = {k: float(base[k]) for k in ("flops_sparse_m", "latency_cpu_ms", "latency_gpu_ms")}
    ft = [r for r in rows if r["status"] == "finetuned"]
    xs = sorted({float(r["sparsity_target"]) for r in ft})

    def series(method, key):
        d = {float(r["sparsity_target"]): float(r[key]) for r in ft if r["method"] == method}
        return [100.0] + [d[x] / b[key] * 100 for x in xs]

    def mean_latency(key):
        return [100.0] + [sum(float(r[key]) for r in ft if float(r["sparsity_target"]) == x)
                          / sum(1 for r in ft if float(r["sparsity_target"]) == x) / b[key] * 100
                          for x in xs]

    X = [0.0] + xs
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    flops_style = {"local": ("#9a9893", "-"), "global": ("#9a9893", "--")}
    for m in ("local", "global"):
        c, ls = flops_style[m]
        ys = series(m, "flops_sparse_m")
        ax.plot(X, ys, color=c, linestyle=ls, linewidth=2, marker="o", markersize=4,
                label=f"Sparse FLOPs - {m}")
        ax.annotate(f"{ys[-1]:.0f}%", (X[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=INK2)
    # aqua / violet: blue & orange already mean local / global in the other figures
    for key, label, c, dy in (("latency_cpu_ms", "Latency CPU (mean of local/global)", "#1baf7a", -8),
                              ("latency_gpu_ms", "Latency GPU (mean of local/global)", "#6250d6", 8)):
        ys = mean_latency(key)
        ax.plot(X, ys, color=c, linewidth=2.5, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax.annotate(f"{ys[-1]:.0f}%", (X[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, dy), va="center", fontsize=9, color=INK)
    ax.axhline(100, color=INK2, linestyle=":", linewidth=1)
    pct_ticks(ax, X)
    ax.set_ylim(0, 120)
    ax.set_xlim(-0.04, 0.98)
    style(ax, "Q4  FLOPs drop up to 90%, latency stays ~100% of baseline",
          "% of baseline (baseline = 100%)")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    return fig


def main():
    baseline, data = load()
    os.makedirs(OUT_DIR, exist_ok=True)
    figs = {
        "q1_q2_accuracy_before_finetune.png": plot_before(baseline, data),
        "q3_accuracy_after_finetune.png": plot_after(baseline, data),
        "q3_recovery.png": plot_recovery(baseline, data),
    }
    if os.path.exists(EFF_PATH):
        figs["q4_flops_vs_latency.png"] = plot_q4()
    else:
        print("skip Q4: results/efficiency.csv not found")
    for name, fig in figs.items():
        fig.tight_layout()
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("saved", os.path.relpath(path, REPO_ROOT))


if __name__ == "__main__":
    main()
