"""Q5: channel (structured) pruning -> fine-tune, with the SAME recipe as Q3.

Uses pruning/structured.py (Hoa) to physically remove channels, then fine-tunes exactly like
training/run_pruning.py (SGD + Nesterov, lr 0.01 -> cosine, 10 epochs, wd 5e-4, AMP), so the
accuracies are directly comparable with Q3 and with the control run (finetune_control.py).

    # seed 42 (main run): checkpoints go to Drive so benchmark/measure.py can time them
    python training/finetune_structured.py --ckpt-dir {CKPT}
    # extra seeds: baseline from checkpoints_seed{N}, pruned checkpoints only in /content
    python training/finetune_structured.py --ckpt-dir {CK_SEED} --save-dir /content/struct_seed1 --seed 1

Saves   resnet18_structured_{ratio}_{raw|finetuned}.pt  (save_structured format: widths + weights,
        load with pruning.structured.load_structured; benchmark/measure.py detects it)
Writes  results/accuracy_structured.csv (seed 42) or results/accuracy_structured_seed{N}.csv
        columns: method=structured, sparsity=channel ratio, acc_before/after_finetune, seed
Skips ratios already finished (row in CSV + finetuned file present). ~3 min per ratio on a T4.
"""
import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from pruning.structured import channel_prune, count_params, save_structured  # noqa: E402
from training.common import (SEED, build_model, evaluate, get_device, get_loaders,  # noqa: E402
                             make_scaler, read_rows, set_seed, train_one_epoch, upsert_row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--ckpt-dir", default="/content/drive/MyDrive/pruning-project/checkpoints",
                   help="folder that holds resnet18_baseline.pt")
    p.add_argument("--save-dir", default=None, help="where to save pruned models (default: --ckpt-dir)")
    p.add_argument("--ratios", default="0.25,0.5,0.75")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--results", default=None,
                   help="default results/accuracy_structured.csv (seed 42) or _seed{N}.csv")
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--ft-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--force", action="store_true", help="redo ratios that are already done")
    p.add_argument("--smoke", action="store_true", help="fake data, 1 epoch, random baseline")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.ft_epochs = 1
    save_dir = args.save_dir or args.ckpt_dir
    os.makedirs(save_dir, exist_ok=True)
    results = args.results or os.path.join(
        REPO_ROOT, "results",
        "accuracy_structured.csv" if args.seed == SEED else f"accuracy_structured_seed{args.seed}.csv")
    ratios = [round(float(r), 2) for r in args.ratios.split(",")]

    device = get_device()
    torch.backends.cudnn.benchmark = True
    base = build_model()
    bpath = os.path.join(args.ckpt_dir, "resnet18_baseline.pt")
    if os.path.exists(bpath):
        base.load_state_dict(torch.load(bpath, map_location="cpu"))
    elif not args.smoke:
        sys.exit(f"Missing {bpath}")
    base.eval()

    set_seed(args.seed)
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers,
                                            smoke=args.smoke, seed=args.seed)
    base_acc = evaluate(base.to(device), test_loader, device)
    print(f"Baseline accuracy: {base_acc:.2f}%  (seed {args.seed})", flush=True)
    upsert_row(results, {"method": "baseline", "sparsity": 0.0, "acc_before_finetune": f"{base_acc:.2f}",
                         "acc_after_finetune": f"{base_acc:.2f}", "seed": args.seed})
    base.cpu()
    done = {round(float(r["sparsity"]), 2) for r in read_rows(results) if r["method"] == "structured"}

    for r in ratios:
        ft_path = os.path.join(save_dir, f"resnet18_structured_{r}_finetuned.pt")
        if r in done and os.path.exists(ft_path) and not args.force:
            print(f"[skip] structured {r} already done", flush=True)
            continue
        t0 = time.time()
        model = channel_prune(base, r)
        params = count_params(model) / 1e6
        model.to(device)
        raw = evaluate(model, test_loader, device)
        save_structured(model.cpu(), os.path.join(save_dir, f"resnet18_structured_{r}_raw.pt"))
        model.to(device)

        set_seed(args.seed)  # same seed for every ratio, like run_pruning.py
        opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ft_epochs)
        scaler = make_scaler(device)
        for ep in range(args.ft_epochs):
            loss, _ = train_one_epoch(model, train_loader, opt, scaler, device)
            sched.step()
            print(f"  structured {r} ft {ep + 1}/{args.ft_epochs} loss={loss:.4f}", flush=True)
        ft = evaluate(model, test_loader, device)
        save_structured(model.cpu(), ft_path)
        upsert_row(results, {"method": "structured", "sparsity": r, "acc_before_finetune": f"{raw:.2f}",
                             "acc_after_finetune": f"{ft:.2f}", "seed": args.seed})
        print(f"[done] structured {r}: params {params:.2f}M  raw={raw:.2f}%  finetuned={ft:.2f}%  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\nResults: {os.path.relpath(results, REPO_ROOT)}")
    with open(results) as f:
        print(f.read())


if __name__ == "__main__":
    main()
