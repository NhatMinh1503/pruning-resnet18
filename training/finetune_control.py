"""Control run for Q3: fine-tune the UNPRUNED baseline with the same recipe as run_pruning.py.

Question it answers: "the pruned models got 10 extra epochs - is the comparison fair?"
If the baseline itself gains/loses accuracy with 10 more epochs at lr 0.01, the pruned
results should be compared against that number too.

    python training/finetune_control.py --ckpt-dir /content/drive/MyDrive/pruning-project/checkpoints

Writes one row (method = baseline_ft10, sparsity = 0.0) to --results
(default results/accuracy_control.csv) and saves resnet18_baseline_ft10.pt in a
_control/ subfolder so benchmark/ never picks it up. ~5 min on a Colab T4.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.common import (SEED, build_model, evaluate, get_device, get_loaders,  # noqa: E402
                             make_scaler, set_seed, train_one_epoch, upsert_row)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--ckpt-dir", default="/content/drive/MyDrive/pruning-project/checkpoints")
    p.add_argument("--results", default=os.path.join(REPO_ROOT, "results", "accuracy_control.csv"))
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--ft-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--smoke", action="store_true", help="1 epoch on fake data")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.ft_epochs = 1
    device = get_device()
    torch.backends.cudnn.benchmark = True
    baseline_path = os.path.join(args.ckpt_dir, "resnet18_baseline.pt")
    if not os.path.exists(baseline_path) and not args.smoke:
        sys.exit(f"Missing {baseline_path}")

    set_seed(args.seed)  # same seed as every pruning run
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers,
                                            smoke=args.smoke, seed=args.seed)
    model = build_model()
    if os.path.exists(baseline_path):
        model.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    model.to(device)
    acc_before = evaluate(model, test_loader, device)

    # identical recipe to run_pruning.py, only without masks
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs)
    scaler = make_scaler(device)
    t0 = time.time()
    for ep in range(args.ft_epochs):
        loss, _ = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        print(f"  baseline ft {ep + 1}/{args.ft_epochs} loss={loss:.4f}", flush=True)
    acc_after = evaluate(model, test_loader, device)

    out_dir = os.path.join(args.ckpt_dir, "_control")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "resnet18_baseline_ft10.pt"))
    upsert_row(args.results, {"method": "baseline_ft10", "sparsity": 0.0,
                              "acc_before_finetune": f"{acc_before:.2f}",
                              "acc_after_finetune": f"{acc_after:.2f}", "seed": args.seed})
    print(f"\nbaseline {acc_before:.2f}% -> +{args.ft_epochs} ep (lr {args.ft_lr}) "
          f"{acc_after:.2f}%  ({time.time() - t0:.0f}s)  -> {args.results}")


if __name__ == "__main__":
    main()
