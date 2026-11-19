"""Q5: Structured (channel) pruning -> fine-tune -> accuracy + real speed.

Owner: Hoa.

For every ratio: channel_prune(baseline, ratio) -> eval -> fine-tune -> eval,
then measure params, file size, FLOPs and latency with the SAME functions as
benchmark/measure.py, so the numbers can be compared with efficiency.csv.

Output: results/structured.csv
    method, ratio, params_m, file_size_mb, flops_m, latency_cpu_ms, latency_gpu_ms,
    acc_before_finetune, acc_after_finetune, seed

Usage
-----
    # full run on Colab (GPU), same fine-tune settings as training/run_pruning.py
    python pruning/run_structured.py --checkpoint {CK}/resnet18_baseline.pt \
        --data-dir /content/data --ckpt-dir {CK}

    # speed only, no dataset and no checkpoint (accuracy columns stay empty)
    python pruning/run_structured.py --random-init

    # pipeline test on CPU with fake data
    python pruning/run_structured.py --random-init --smoke
"""
import argparse
import csv
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from benchmark.measure import measure_flops, measure_latency  # noqa: E402
from pruning.structured import channel_prune, count_params, save_structured  # noqa: E402
from training.common import (SEED, build_model, evaluate, get_device,  # noqa: E402
                             get_loaders, load_weights, make_scaler, set_seed,
                             train_one_epoch)

CSV_HEADER = ["method", "ratio", "params_m", "file_size_mb", "flops_m", "latency_cpu_ms",
              "latency_gpu_ms", "acc_before_finetune", "acc_after_finetune", "seed"]


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to resnet18_baseline.pt")
    src.add_argument("--random-init", action="store_true",
                     help="untrained model: speed numbers only, no accuracy")
    p.add_argument("--ratios", default="0.25,0.5,0.75")
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--ckpt-dir", default=os.path.join(REPO_ROOT, "checkpoints"))
    p.add_argument("--csv", default=os.path.join(REPO_ROOT, "results", "structured.csv"))
    p.add_argument("--ft-epochs", type=int, default=10, help="0 = no fine-tune (raw accuracy only)")
    p.add_argument("--ft-lr", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="fake data, 1 epoch")
    return p.parse_args()


def file_size_mb(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_structured(model, path)
    return os.path.getsize(path) / (1024 * 1024)


def speed(model):
    """Params, FLOPs and latency; model is returned on CPU."""
    model = model.cpu().eval()
    row = {"params_m": f"{count_params(model) / 1e6:.3f}",
           "flops_m": f"{measure_flops(model):.2f}",
           "latency_cpu_ms": f"{measure_latency(model, 'cpu'):.3f}",
           "latency_gpu_ms": ""}
    if torch.cuda.is_available():
        row["latency_gpu_ms"] = f"{measure_latency(model, 'cuda'):.3f}"
    model.cpu()
    return row


def fine_tune(model, train_loader, device, epochs, lr):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = make_scaler(device)
    for ep in range(epochs):
        loss, acc = train_one_epoch(model, train_loader, opt, scaler, device)
        sched.step()
        print(f"    ft {ep + 1}/{epochs} loss={loss:.4f} train_acc={acc:.2f}%", flush=True)
    return model


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    ratios = [round(float(r), 2) for r in args.ratios.split(",")]
    if args.smoke:
        args.ft_epochs = 1

    base = build_model()
    if args.checkpoint:
        load_weights(base, args.checkpoint)
    base.eval()

    with_acc = bool(args.checkpoint) or args.smoke
    if with_acc:
        train_loader, test_loader = get_loaders(args.data_dir, num_workers=args.num_workers,
                                                smoke=args.smoke, seed=SEED)

    rows = []
    for r in [0.0] + ratios:
        tag = "baseline" if r == 0.0 else f"structured_{r}"
        print(f"[{tag}]", flush=True)
        model = base if r == 0.0 else channel_prune(base, r)
        row = {"method": "baseline" if r == 0.0 else "structured", "ratio": r, "seed": SEED,
               "acc_before_finetune": "", "acc_after_finetune": ""}
        if with_acc:
            row["acc_before_finetune"] = f"{evaluate(model.to(device), test_loader, device):.2f}"
            if r == 0.0:
                row["acc_after_finetune"] = row["acc_before_finetune"]
            elif args.ft_epochs > 0:
                set_seed(SEED)
                fine_tune(model, train_loader, device, args.ft_epochs, args.ft_lr)
                row["acc_after_finetune"] = f"{evaluate(model, test_loader, device):.2f}"
        stage = "finetuned" if (with_acc and r > 0 and args.ft_epochs > 0) else "raw"
        path = os.path.join(args.ckpt_dir, f"resnet18_{tag}_{stage}.pt")
        row["file_size_mb"] = f"{file_size_mb(model.cpu(), path):.2f}"
        row.update(speed(model))
        rows.append(row)
        print("   ", {k: row[k] for k in CSV_HEADER}, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
