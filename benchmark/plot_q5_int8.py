"""Generate publication-ready figures for Q5 (Structured Pruning Multi-batch Latency)
and INT8 Post-Training Quantization (Storage, CPU Speedup, Accuracy & Pareto Frontier).

Owner: Tran Quang Canh (benchmark/).
Reads:
  - results/structured_efficiency.csv
  - results/int8_efficiency.csv
Writes:
  - results/figures/q5_multibatch_latency.png
  - results/figures/q5_int8_quantization.png
  - results/figures/q5_int8_pareto_frontier.png

Usage:
  python benchmark/plot_q5_int8.py
"""
import argparse
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STRUCT_CSV = os.path.join(REPO_ROOT, "results", "structured_efficiency.csv")
DEFAULT_INT8_CSV = os.path.join(REPO_ROOT, "results", "int8_efficiency.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "figures")

# Team standard color palette & editorial surface tokens
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

COLOR_BASE = "#52514e"
COLOR_LOCAL = "#2a78d6"
COLOR_GLOBAL = "#eb6834"
COLOR_STRUCT = "#1baf7a"
COLOR_STRUCT_25 = "#68d3ab"
COLOR_STRUCT_50 = "#1baf7a"
COLOR_STRUCT_75 = "#0e7550"

METHOD_COLORS = {
    "baseline": "#0b0b0b",
    "local": COLOR_LOCAL,
    "global": COLOR_GLOBAL,
    "structured": COLOR_STRUCT,
}


def apply_editorial_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent editorial spine, grid, and background styling."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK2, labelpad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.85)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def get_row(df: pd.DataFrame, method: str, sparsity: float, status: str) -> pd.Series:
    """Select a single row from DataFrame matching (method, sparsity_target, status)."""
    sub = df[
        (df["method"] == method)
        & (np.isclose(df["sparsity_target"].astype(float), sparsity))
        & (df["status"] == status)
    ]
    if sub.empty:
        raise ValueError(f"Missing row for method={method}, sparsity={sparsity}, status={status}")
    return sub.iloc[0]


def plot_q5_multibatch_latency(struct_csv: str, out_dir: str) -> str:
    """Figure 1: Multi-batch CPU & GPU Latency (% of Baseline) across BS=1, 16, 64."""
    df = pd.read_csv(struct_csv)
    base = get_row(df, "baseline", 0.0, "baseline")

    specs = [
        ("local", 0.9, "finetuned", "Local 90% (unstructured)", COLOR_LOCAL),
        ("global", 0.9, "finetuned", "Global 90% (unstructured)", COLOR_GLOBAL),
        ("structured", 0.25, "finetuned", "Structured 25% (channels)", COLOR_STRUCT_25),
        ("structured", 0.5, "finetuned", "Structured 50% (channels)", COLOR_STRUCT_50),
        ("structured", 0.75, "finetuned", "Structured 75% (channels)", COLOR_STRUCT_75),
    ]
    rows = [get_row(df, m, s, st) for m, s, st, _, _ in specs]
    batch_sizes = [1, 16, 64]

    fig, (ax_cpu, ax_gpu) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    x = np.arange(len(batch_sizes))
    n_bars = len(specs)
    width = 0.145

    for dev_idx, (ax, dev_key, dev_title, base_fmt) in enumerate([
        (ax_cpu, "cpu", "(a) CPU Latency — Intel Xeon 2.00GHz (2 vCPU)", "{:.1f}"),
        (ax_gpu, "gpu", "(b) GPU Latency — NVIDIA Tesla T4 (15 GB)", "{:.2f}"),
    ]):
        base_lats = [float(base[f"latency_{dev_key}_bs{bs}_ms"]) for bs in batch_sizes]

        for i, (row, (_, sp, _, label, color)) in enumerate(zip(rows, specs)):
            offset = (i - (n_bars - 1) / 2.0) * width
            lats = [float(row[f"latency_{dev_key}_bs{bs}_ms"]) for bs in batch_sizes]
            norm_pcts = [lat / b_lat * 100.0 for lat, b_lat in zip(lats, base_lats)]

            bars = ax.bar(
                x + offset,
                norm_pcts,
                width=width - 0.012,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.8,
                label=label if dev_idx == 0 else None,
            )

            # Annotate speedup on BS=64 for Structured 50% and 75%
            if sp in (0.5, 0.75) and row["method"] == "structured":
                bs64_lat = lats[2]
                bs64_base = base_lats[2]
                speedup = bs64_base / bs64_lat
                bar_x = x[2] + offset
                bar_y = norm_pcts[2]
                ax.text(
                    bar_x,
                    bar_y + 2.2,
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                    color=INK,
                )

        # Baseline 100% reference line
        ax.axhline(100.0, color=INK2, linestyle="--", linewidth=1.1)
        ax.text(
            2.46,
            101.8,
            "Baseline = 100%",
            fontsize=8.2,
            color=INK2,
            ha="right",
            va="bottom",
        )

        x_labels = [
            f"BS={bs}\n(Base: {base_fmt.format(b_lat)} ms)"
            for bs, b_lat in zip(batch_sizes, base_lats)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0, 144)
        ax.set_xlim(-0.5, 2.52)
        apply_editorial_style(
            ax,
            dev_title,
            "Batch Size (with Baseline absolute latency)",
            "Normalized Latency (% of Baseline)",
        )

    # Specific hardware annotations on GPU panel
    gpu_base_bs16 = float(base["latency_gpu_bs16_ms"])
    gpu_str25_bs16 = float(rows[2]["latency_gpu_bs16_ms"])
    gpu_str25_pct16 = gpu_str25_bs16 / gpu_base_bs16 * 100.0
    ax_gpu.annotate(
        f"{gpu_str25_pct16:.0f}% ({gpu_str25_bs16:.2f} ms)\nnon-pow2 channels",
        xy=(x[1], gpu_str25_pct16),
        xytext=(x[1] + 0.16, 131.0),
        ha="left",
        va="center",
        fontsize=8.2,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0, connectionstyle="arc3,rad=-0.12"),
    )

    gpu_bs1_min = min(float(r["latency_gpu_bs1_ms"]) for r in rows)
    gpu_bs1_max = max(float(r["latency_gpu_bs1_ms"]) for r in rows)
    ax_gpu.text(
        x[0],
        122.0,
        f"Kernel launch\noverhead ({gpu_bs1_min:.1f}–{gpu_bs1_max:.1f} ms)",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=INK2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor=GRID, lw=0.8),
    )

    handles, labels = ax_cpu.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(specs),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Q5  Multi-batch Inference Latency: Unstructured vs. Structured Channel Pruning",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.015,
        y=0.985,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.89))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q5_multibatch_latency.png")
    fig.savefig(out_path, facecolor=SURFACE, dpi=200)
    plt.close(fig)
    return out_path


def plot_q5_int8_quantization(int8_csv: str, out_dir: str) -> str:
    """Figure 2: INT8 PTQ Impact on Storage (.pt/.zip), CPU BS=64 Latency, and Top-1 Accuracy."""
    df = pd.read_csv(int8_csv)

    order = [
        ("baseline", 0.0, "Base"),
        ("local", 0.5, "Loc .5"),
        ("local", 0.9, "Loc .9"),
        ("global", 0.5, "Glob .5"),
        ("global", 0.9, "Glob .9"),
        ("structured", 0.25, "Str .25"),
        ("structured", 0.5, "Str .5"),
        ("structured", 0.75, "Str .75"),
    ]
    ordered_rows = []
    short_labels = []
    for m, s, lab in order:
        st = "baseline" if m == "baseline" else "finetuned"
        ordered_rows.append(get_row(df, m, s, st))
        short_labels.append(lab)

    df_ord = pd.DataFrame(ordered_rows).reset_index(drop=True)
    x = np.arange(len(df_ord))
    w = 0.36

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # --- Panel 1: Model Storage (.pt outer vs .zip inner) ---
    fp32_pt = df_ord["fp32_size_mb"].astype(float).values
    fp32_zip = df_ord["fp32_zip_mb"].astype(float).values
    int8_pt = df_ord["int8_size_mb"].astype(float).values
    int8_zip = df_ord["int8_zip_mb"].astype(float).values

    # Left position: FP32 (.pt light gray outer, .zip dark gray inner)
    ax1.bar(
        x - w / 2,
        fp32_pt,
        width=w - 0.03,
        color="#d5d3ce",
        edgecolor="#7c7a75",
        linewidth=0.9,
        label="FP32 .pt (raw)",
    )
    ax1.bar(
        x - w / 2,
        fp32_zip,
        width=(w - 0.03) * 0.68,
        color="#52514e",
        edgecolor="none",
        label="FP32 .zip",
    )

    # Right position: INT8 (.pt light aqua outer, .zip dark aqua inner)
    ax1.bar(
        x + w / 2,
        int8_pt,
        width=w - 0.03,
        color="#9fe5cb",
        edgecolor="#1baf7a",
        linewidth=0.9,
        label="INT8 .pt (raw)",
    )
    ax1.bar(
        x + w / 2,
        int8_zip,
        width=(w - 0.03) * 0.68,
        color="#0e7550",
        edgecolor="none",
        label="INT8 .zip",
    )

    # Highlight Global 90% INT8 (.zip) and Struct 75% INT8 (.pt)
    glob9_idx = 4
    str75_idx = 7
    ax1.annotate(
        f"Glob .9 INT8:\n{int8_zip[glob9_idx]:.2f} MB (.zip)",
        xy=(x[glob9_idx] + w / 2, int8_zip[glob9_idx]),
        xytext=(5.85, 34.5),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9, connectionstyle="arc3,rad=-0.15"),
    )
    ax1.annotate(
        f"Str .75 INT8:\n{int8_pt[str75_idx]:.2f} MB (.pt)",
        xy=(x[str75_idx] + w / 2, int8_pt[str75_idx]),
        xytext=(6.75, 21.5),
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="#0e7550",
        arrowprops=dict(arrowstyle="->", color="#0e7550", lw=1.0, connectionstyle="arc3,rad=-0.12"),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=30, ha="right")
    ax1.set_ylim(0, 50)
    apply_editorial_style(ax1, "(a) Disk Storage: .pt vs. .zip (MB)", "Model Configuration", "File Size (MB)")
    ax1.legend(frameon=False, loc="upper right", fontsize=8.0, ncol=2)

    # --- Panel 2: CPU Latency at BS=64 (ms) & Speedup ---
    fp32_lat64 = df_ord["fp32_lat_bs64_ms"].astype(float).values
    int8_lat64 = df_ord["int8_lat_bs64_ms"].astype(float).values
    speedups64 = fp32_lat64 / int8_lat64

    ax2.bar(
        x - w / 2,
        fp32_lat64,
        width=w - 0.03,
        color="#c8c6c0",
        edgecolor="#6e6c67",
        linewidth=0.8,
        label="FP32 Latency (CPU)",
    )
    bar_colors = [METHOD_COLORS[m] for m in df_ord["method"]]
    ax2.bar(
        x + w / 2,
        int8_lat64,
        width=w - 0.03,
        color=bar_colors,
        edgecolor=SURFACE,
        linewidth=0.8,
        label="INT8 Latency (onednn)",
    )

    for i in range(len(df_ord)):
        ax2.text(
            x[i] + w / 2,
            int8_lat64[i] + 14.0,
            f"{speedups64[i]:.1f}x",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=INK,
        )

    total_speedup_str75 = fp32_lat64[0] / int8_lat64[str75_idx]
    ax2.annotate(
        f"{int8_lat64[str75_idx]:.1f} ms\n({total_speedup_str75:.1f}x vs Base FP32)",
        xy=(x[str75_idx] + w / 2, int8_lat64[str75_idx] + 50.0),
        xytext=(6.6, 515.0),
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#0e7550",
        arrowprops=dict(arrowstyle="->", color="#0e7550", lw=1.0, connectionstyle="arc3,rad=-0.12"),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=30, ha="right")
    ax2.set_ylim(0, 880)
    apply_editorial_style(ax2, "(b) CPU Latency at BS=64 (ms)", "Model Configuration", "Inference Latency (ms)")
    ax2.legend(frameon=False, loc="upper right", fontsize=8.2)

    # --- Panel 3: Top-1 Accuracy Dumbbell / Drop Chart ---
    fp32_acc = df_ord["fp32_acc"].astype(float).values
    int8_acc = df_ord["int8_acc"].astype(float).values
    acc_drop = fp32_acc - int8_acc

    for i in range(len(df_ord)):
        c = METHOD_COLORS[df_ord.loc[i, "method"]]
        ax3.vlines(x[i], int8_acc[i], fp32_acc[i], color=c, linestyle="-", linewidth=2.0, alpha=0.75, zorder=2)
        ax3.scatter(
            x[i],
            fp32_acc[i],
            color=c,
            marker="o",
            s=52,
            edgecolors=SURFACE,
            linewidths=1.2,
            zorder=3,
        )
        ax3.scatter(
            x[i],
            int8_acc[i],
            color=c,
            marker="D",
            s=46,
            edgecolors=SURFACE,
            linewidths=1.2,
            zorder=3,
        )
        ax3.text(
            x[i],
            int8_acc[i] - 0.16,
            f"-{acc_drop[i]:.2f}",
            ha="center",
            va="top",
            fontsize=7.8,
            fontweight="bold",
            color=c,
        )

    ax3.axhline(fp32_acc[0], color=INK2, linestyle="--", linewidth=0.9, alpha=0.7)
    legend_handles_p3 = [
        Line2D([0], [0], marker="o", color="w", label="FP32 Accuracy", markerfacecolor=INK, markersize=7),
        Line2D([0], [0], marker="D", color="w", label="INT8 Accuracy (PTQ)", markerfacecolor=INK, markersize=6.5),
    ]
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_labels, rotation=30, ha="right")
    ax3.set_ylim(91.8, 95.6)
    apply_editorial_style(ax3, "(c) Top-1 Test Accuracy & PTQ Drop (%)", "Model Configuration", "Test Accuracy (%)")
    ax3.legend(handles=legend_handles_p3, frameon=False, loc="lower left", fontsize=8.2)

    fig.suptitle(
        "INT8 Post-Training Quantization (PyTorch FX, onednn): Storage, CPU Latency & Accuracy Drop",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.012,
        y=0.98,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q5_int8_quantization.png")
    fig.savefig(out_path, facecolor=SURFACE, dpi=200)
    plt.close(fig)
    return out_path


def compute_pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Compute Pareto-optimal points for (minimize X, maximize Y)."""
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: List[Tuple[float, float]] = []
    max_y = -float("inf")
    for px, py in sorted_pts:
        if py > max_y:
            frontier.append((px, py))
            max_y = py
    return frontier


def plot_q5_int8_pareto_frontier(int8_csv: str, out_dir: str) -> str:
    """Figure 3: Unified Pareto Frontier (Accuracy vs Raw .pt Size & CPU BS=64 Latency)."""
    df = pd.read_csv(int8_csv)

    records = []
    for _, r in df.iterrows():
        m = r["method"]
        sp = float(r["sparsity_target"])
        tag = "Base" if m == "baseline" else f"{m.capitalize()[:4]}.{int(round(sp * 100))}"
        if m == "structured":
            tag = f"Str.{int(round(sp * 100))}"
        elif m == "global":
            tag = f"Glob.{int(round(sp * 100))}"
        elif m == "local":
            tag = f"Loc.{int(round(sp * 100))}"

        records.append({
            "key": f"{tag}_FP32",
            "method": m,
            "prec": "FP32",
            "size_mb": float(r["fp32_size_mb"]),
            "lat_ms": float(r["fp32_lat_bs64_ms"]),
            "acc": float(r["fp32_acc"]),
        })
        records.append({
            "key": f"{tag}_INT8",
            "method": m,
            "prec": "INT8",
            "size_mb": float(r["int8_size_mb"]),
            "lat_ms": float(r["int8_lat_bs64_ms"]),
            "acc": float(r["int8_acc"]),
        })

    pts_df = pd.DataFrame(records)

    fig, (ax_size, ax_lat) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    # Plot points & Pareto frontier for both panels
    for ax, x_col, title, xlabel, xlim in [
        (ax_size, "size_mb", "(a) Accuracy vs. Raw Model Size (.pt, MB)", "Raw Model File Size on Disk (.pt, MB)  [← Smaller is better]", (-7.5, 47.0)),
        (ax_lat, "lat_ms", "(b) Accuracy vs. CPU Latency at BS=64 (ms)", "CPU Inference Latency at BS=64 (ms)  [← Faster is better]", (65.0, 845.0)),
    ]:
        pairs = list(zip(pts_df[x_col].values, pts_df["acc"].values))
        frontier = compute_pareto_frontier(pairs)
        fx, fy = zip(*frontier)
        ax.plot(
            fx,
            fy,
            color=INK,
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            zorder=2,
            label="Pareto Frontier",
        )

        for _, row in pts_df.iterrows():
            c = METHOD_COLORS[row["method"]]
            mk = "o" if row["prec"] == "FP32" else "D"
            sz = 64 if row["prec"] == "FP32" else 56
            ax.scatter(
                row[x_col],
                row["acc"],
                color=c,
                marker=mk,
                s=sz,
                edgecolors=SURFACE,
                linewidths=1.3,
                zorder=4,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(92.45, 95.45)
        apply_editorial_style(ax, title, xlabel, "Top-1 Test Accuracy (%)" if ax is ax_size else "")

    # Careful non-overlapping label offsets for Panel 1 (Size MB)
    size_annotations = {
        "Base_FP32": ("Baseline FP32", (-8, 5), "right"),
        "Glob.90_FP32": ("Glob.9 FP32", (-6, -12), "right"),
        "Loc.90_FP32": ("Loc.9 FP32", (-8, -2), "right"),
        "Str.25_FP32": ("Str.25 FP32", (-6, 9), "right"),
        "Str.50_FP32": ("Str.50 FP32", (-6, 9), "right"),
        "Str.75_FP32": ("Str.75 FP32", (8, -3), "left"),
        "Glob.50_INT8": ("Base / Glob.5 INT8", (10, -1), "left"),
        "Glob.90_INT8": ("Glob.9 INT8", (-8, -4), "right"),
        "Loc.90_INT8": ("Loc.9 INT8", (8, -3), "left"),
        "Str.25_INT8": ("Str.25 INT8", (-8, 4), "right"),
        "Str.50_INT8": ("Str.50 INT8", (-8, 3), "right"),
        "Str.75_INT8": ("Str.75 INT8", (8, -3), "left"),
    }
    for key, (text, offset, ha) in size_annotations.items():
        row = pts_df[pts_df["key"] == key].iloc[0]
        ax_size.annotate(
            f"{text} ({row['size_mb']:.1f}MB)",
            (row["size_mb"], row["acc"]),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va="center",
            fontsize=7.9,
            fontweight="bold" if "Str" in key or "Base_FP32" in key else "normal",
            color=METHOD_COLORS[row["method"]],
        )

    # Careful non-overlapping label offsets for Panel 2 (CPU Latency BS=64 ms)
    lat_annotations = {
        "Base_FP32": ("Baseline FP32", (-6, 8), "right"),
        "Glob.50_FP32": ("Glob.5 FP32", (-8, 5), "right"),
        "Glob.90_FP32": ("Glob.9 FP32", (4, -14), "right"),
        "Loc.90_FP32": ("Loc.9 FP32", (-8, -2), "right"),
        "Str.25_FP32": ("Str.25 FP32", (-8, 6), "right"),
        "Str.50_FP32": ("Str.50 FP32", (9, -6), "left"),
        "Str.75_FP32": ("Str.75 FP32", (-8, 5), "right"),
        "Glob.50_INT8": ("Base / Glob.5 INT8", (-10, 5), "right"),
        "Glob.90_INT8": ("Glob.9 INT8", (8, -6), "left"),
        "Loc.90_INT8": ("Loc.9 INT8", (8, -2), "left"),
        "Str.25_INT8": ("Str.25 INT8", (8, -4), "left"),
        "Str.50_INT8": ("Str.50 INT8", (8, -3), "left"),
        "Str.75_INT8": ("Str.75 INT8", (8, 2), "left"),
    }
    for key, (text, offset, ha) in lat_annotations.items():
        row = pts_df[pts_df["key"] == key].iloc[0]
        ax_lat.annotate(
            f"{text} ({row['lat_ms']:.0f}ms)",
            (row["lat_ms"], row["acc"]),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va="center",
            fontsize=7.9,
            fontweight="bold" if "Str" in key or "Base_FP32" in key else "normal",
            color=METHOD_COLORS[row["method"]],
        )

    legend_elements = [
        Patch(facecolor=METHOD_COLORS["baseline"], label="Baseline"),
        Patch(facecolor=METHOD_COLORS["local"], label="Local (Unstructured)"),
        Patch(facecolor=METHOD_COLORS["global"], label="Global (Unstructured)"),
        Patch(facecolor=METHOD_COLORS["structured"], label="Structured (Channels)"),
        Line2D([0], [0], marker="o", color="w", label="FP32 Model", markerfacecolor=INK, markersize=7.5),
        Line2D([0], [0], marker="D", color="w", label="INT8 Quantized (PTQ)", markerfacecolor=INK, markersize=7),
        Line2D([0], [0], color=INK, linestyle="--", lw=1.5, label="Pareto Frontier"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=8.6,
    )
    fig.suptitle(
        "Unified Compression Pareto Frontier: Structured Pruning + INT8 Quantization (ResNet-18 / CIFAR-10)",
        fontsize=13,
        fontweight="bold",
        color=INK,
        x=0.015,
        y=0.985,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.89))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q5_int8_pareto_frontier.png")
    fig.savefig(out_path, facecolor=SURFACE, dpi=200)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Q5 multi-batch latency and INT8 quantization figures.")
    parser.add_argument("--structured-csv", default=DEFAULT_STRUCT_CSV, help="Path to structured_efficiency.csv")
    parser.add_argument("--int8-csv", default=DEFAULT_INT8_CSV, help="Path to int8_efficiency.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    f1 = plot_q5_multibatch_latency(args.structured_csv, args.out_dir)
    print(f"[OK] Saved Figure 1: {os.path.relpath(f1, REPO_ROOT)}")

    f2 = plot_q5_int8_quantization(args.int8_csv, args.out_dir)
    print(f"[OK] Saved Figure 2: {os.path.relpath(f2, REPO_ROOT)}")

    f3 = plot_q5_int8_pareto_frontier(args.int8_csv, args.out_dir)
    print(f"[OK] Saved Figure 3: {os.path.relpath(f3, REPO_ROOT)}")


if __name__ == "__main__":
    main()
