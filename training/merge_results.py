"""Merge results/accuracy.csv (19 rows) + results/efficiency.csv (37 rows)
-> results/merged_results.csv (37 rows, one per checkpoint).

    python training/merge_results.py

Join rule (efficiency.csv is the base):
    status = baseline   -> accuracy of the 'baseline' row
    status = raw        -> acc_before_finetune  (same method + sparsity)
    status = finetuned  -> acc_after_finetune   (same method + sparsity)
"""
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(REPO_ROOT, "results", "accuracy.csv")
EFF = os.path.join(REPO_ROOT, "results", "efficiency.csv")
OUT = os.path.join(REPO_ROOT, "results", "merged_results.csv")

COLUMNS = ["checkpoint_name", "method", "sparsity", "status", "accuracy",
           "actual_sparsity", "file_size_mb", "zip_size_mb", "flops_m", "flops_sparse_m",
           "latency_cpu_ms", "latency_gpu_ms"]


def key(method, sparsity):
    return method, round(float(sparsity), 2)


def main():
    acc = {key(r["method"], r["sparsity"]): r for r in csv.DictReader(open(ACC, newline=""))}
    eff = list(csv.DictReader(open(EFF, newline="")))

    rows, missing = [], []
    for e in eff:
        status = e["status"]
        k = key("baseline", 0.0) if status == "baseline" else key(e["method"], e["sparsity_target"])
        a = acc.get(k)
        if a is None:
            missing.append(e["checkpoint_name"])
            continue
        col = "acc_before_finetune" if status == "raw" else "acc_after_finetune"
        rows.append({
            "checkpoint_name": e["checkpoint_name"], "method": e["method"],
            "sparsity": f"{float(e['sparsity_target']):.1f}", "status": status,
            "accuracy": a[col],
            **{c: e[c] for c in COLUMNS[5:]},
        })

    if missing:
        sys.exit(f"No accuracy for: {missing}")
    order = {"baseline": 0, "local": 1, "global": 2}
    rows.sort(key=lambda r: (order.get(r["method"], 9), float(r["sparsity"]), r["status"] != "raw"))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {os.path.relpath(OUT, REPO_ROOT)}")
    if len(rows) != 37:
        print(f"WARNING: expected 37 rows, got {len(rows)}")


if __name__ == "__main__":
    main()
