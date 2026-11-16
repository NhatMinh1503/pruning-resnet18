"""Where do the FLOPs numbers come from? Per-layer breakdown + cross-check of the 2 methods.

    python training/flops_breakdown.py

Method A (thop.profile, used for `flops_m` in efficiency.csv)
    Counts MACs of EVERY module (Conv, Linear, BatchNorm, ReLU, AvgPool, residual add)
    and never looks at the weight values -> 1115.78 MFLOPs for every checkpoint.
Method B (own forward hook, used for `flops_sparse_m`, benchmark/measure.py)
    Counts only Conv2d / Linear:
        Conv2d MACs = weight.numel() x H_out x W_out      (= C_out x C_in x k x k x H_out x W_out)
        Linear MACs = weight.numel()                      (= out x in)
    dense  : all weights           -> 1110.85 MFLOPs
    sparse : only weights != 0     -> e.g. Local 90% 111.08, Global 90% 281.75
Both use FLOPs = 2 x MACs (1 multiply + 1 add), input (1, 3, 32, 32).

Outputs results/flops_breakdown.csv (one row per Conv/Linear layer) and prints a check that
"dense x (1 - layer sparsity)" from results/layer_sparsity.csv reproduces efficiency.csv.
"""
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.common import build_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flops_breakdown.csv")
LAYER_SP = os.path.join(ROOT, "results", "layer_sparsity.csv")
EFF = os.path.join(ROOT, "results", "efficiency.csv")


def per_layer(model):
    rows = []

    def hook(name):
        def f(m, inp, out):
            hw = out.shape[2] * out.shape[3] if isinstance(m, nn.Conv2d) else 1
            rows.append({
                "layer": name,
                "type": "Conv2d" if isinstance(m, nn.Conv2d) else "Linear",
                "weight_shape": "x".join(map(str, m.weight.shape)),
                "out_hw": f"{out.shape[2]}x{out.shape[3]}" if out.dim() == 4 else "-",
                "params": m.weight.numel(),
                "macs": m.weight.numel() * hw,
            })
        return f

    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in hs:
        h.remove()
    total = sum(r["macs"] for r in rows)
    for r in rows:
        r["mflops"] = round(r["macs"] * 2 / 1e6, 3)
        r["flops_share_pct"] = round(r["macs"] / total * 100, 2)
    return rows, total


def thop_total(model):
    try:
        from thop import profile
    except ImportError:
        return None
    import copy
    macs, _ = profile(copy.deepcopy(model), inputs=(torch.zeros(1, 3, 32, 32),), verbose=False)
    return macs * 2 / 1e6


def main():
    model = build_model().eval()
    rows, total_macs = per_layer(model)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    dense_b = total_macs * 2 / 1e6
    print(f"{'layer':24s} {'weight':>14s} {'out':>6s} {'MFLOPs':>9s} {'share':>7s}")
    for r in rows:
        print(f"{r['layer']:24s} {r['weight_shape']:>14s} {r['out_hw']:>6s} "
              f"{r['mflops']:9.2f} {r['flops_share_pct']:6.2f}%")
    stage = {}
    for r in rows:
        key = r["layer"].split(".")[0]
        stage[key] = stage.get(key, 0) + r["flops_share_pct"]
    print("\nFLOPs share by stage:", {k: f"{v:.1f}%" for k, v in stage.items()})

    a = thop_total(model)
    print(f"\nMethod A thop (all modules)      : {a:.2f} MFLOPs" if a else "\nthop not installed")
    print(f"Method B hook (Conv/Linear only) : {dense_b:.2f} MFLOPs")
    if a:
        print(f"difference A - B (BN, ReLU, pool, add): {a - dense_b:.2f} MFLOPs ({(a - dense_b) / a * 100:.2f}%)")

    # Cross-check sparse FLOPs: sum(dense_layer x (1 - sparsity_layer)) vs efficiency.csv
    if os.path.exists(LAYER_SP):
        sp = {}
        for r in csv.DictReader(open(LAYER_SP, newline="")):
            sp[(r["method"], float(r["sparsity"]), r["layer"])] = float(r["layer_sparsity"])
        eff = {}
        if os.path.exists(EFF):
            for r in csv.DictReader(open(EFF, newline="")):
                if r["status"] == "finetuned":
                    eff[(r["method"], float(r["sparsity_target"]))] = float(r["flops_sparse_m"])
        print("\nSparse FLOPs check (from layer sparsity) vs measured (efficiency.csv):")
        for m in ("local", "global"):
            for s in sorted({k[1] for k in sp if k[0] == m}):
                est = sum(r["mflops"] * (1 - sp[(m, s, r["layer"])]) for r in rows)
                meas = eff.get((m, s))
                tail = f"  measured {meas:8.2f}" if meas is not None else ""
                print(f"  {m:6s} {s:.1f}: estimated {est:8.2f}{tail}")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
