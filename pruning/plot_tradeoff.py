"""Accuracy vs effective FLOPs (Q2) and FLOPs vs real latency (Q4/Q5).

Owner: Hoa.  Reads results/accuracy.csv, results/efficiency.csv,
results/structured_speed_prelim.csv and (if present) results/structured.csv; writes results/figures/q2_q5_tradeoff.png.

    python pruning/plot_tradeoff.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    acc = pd.read_csv(os.path.join(R, "accuracy.csv"))
    eff = pd.read_csv(os.path.join(R, "efficiency.csv"))
    eff = eff[eff.status == "finetuned"][["method", "sparsity_target", "flops_sparse_m"]]
    eff = eff.rename(columns={"sparsity_target": "sparsity"})
    base_flops = 1110.85
    df = acc.merge(eff, on=["method", "sparsity"])
    df["flops_pct"] = 100 * df.flops_sparse_m / base_flops

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for method, color in (("local", BLUE), ("global", ORANGE)):
        d = df[df.method == method].sort_values("flops_pct")
        a.plot(d.flops_pct, d.acc_after_finetune, color=color, lw=2, marker="o", ms=5,
               markeredgecolor="white", markeredgewidth=1.5, label=f"{method} (fine-tuned)")
        last = d.iloc[0]
        a.annotate(f"{method} {last.sparsity:.0%}: {last.acc_after_finetune:.2f}%",
                   (last.flops_pct, last.acc_after_finetune), textcoords="offset points",
                   xytext=(8, -4), fontsize=8, color=INK)
    st_path = os.path.join(R, "structured.csv")
    if os.path.exists(st_path):  # written by run_structured.py on Colab
        st = pd.read_csv(st_path)
        st = st[(st.method == "structured") & st.acc_after_finetune.notna()]
        if len(st):
            st = st.sort_values("ratio", ascending=False)
            base_dense = pd.read_csv(st_path).query("method == 'baseline'").flops_m.iloc[0]
            a.plot(100 * st.flops_m / base_dense, st.acc_after_finetune, color=AQUA, lw=2,
                   marker="s", ms=5, markeredgecolor="white", markeredgewidth=1.5,
                   label="structured (fine-tuned)")
    a.axhline(95.10, color=MUTED, lw=1, ls="--")
    a.text(101, 95.12, "baseline 95.10%", fontsize=8, color=MUTED, ha="right", va="bottom")
    a.set_xlabel("Effective FLOPs (% of baseline, zeros skipped)", color=INK, fontsize=9)
    a.set_ylabel("Test accuracy after fine-tune (%)", color=INK, fontsize=9)
    a.set_title("Same sparsity, different FLOPs: Local cuts compute,\nGlobal cuts mostly layer4",
                fontsize=10, color=INK, loc="left")
    a.set_xlim(0, 102)
    a.set_ylim(93.9, 95.3)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    style(a)

    sp = pd.read_csv(os.path.join(R, "structured_speed_prelim.csv"))
    sp["flops_pct"] = 100 * sp.eff_flops_m / sp.eff_flops_m.iloc[0]
    sp["lat_pct"] = 100 * sp.latency_bs64_ms / sp.latency_bs64_ms.iloc[0]
    b.plot([0, 100], [0, 100], color=GRID, lw=1.5, ls="--")
    b.text(30, 25, "ideal: latency follows FLOPs", fontsize=8, color=MUTED, rotation=35)
    for kind, color, lab in (("unstructured", BLUE, "unstructured (mask)"),
                             ("structured", AQUA, "structured (channels removed)")):
        d = pd.concat([sp[sp.kind == "baseline"], sp[sp.kind == kind]]).sort_values("flops_pct")
        b.plot(d.flops_pct, d.lat_pct, color=color, lw=2, marker="o", ms=6,
               markeredgecolor="white", markeredgewidth=1.5, label=lab)
    b.set_xlabel("Effective FLOPs (% of baseline)", color=INK, fontsize=9)
    b.set_ylabel("CPU latency, batch 64 (% of baseline)", color=INK, fontsize=9)
    b.set_title("Only structured pruning turns FLOPs into speed\n(prelim. CPU, 2 threads)",
                fontsize=10, color=INK, loc="left")
    b.set_xlim(0, 105)
    b.set_ylim(0, 115)
    b.legend(frameon=False, fontsize=8, loc="lower right")
    style(b)

    fig.tight_layout()
    out = os.path.join(R, "figures", "q2_q5_tradeoff.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
