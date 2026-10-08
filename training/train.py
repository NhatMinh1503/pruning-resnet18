"""Step 1: train the baseline ResNet-18 on CIFAR-10 and save resnet18_baseline.pt.

Colab (GPU on):
    python training/train.py --ckpt-dir /content/drive/MyDrive/pruning-resnet18/checkpoints

Safe to re-run after a Colab disconnect: it resumes from checkpoints/_resume/baseline_last.pt
(kept out of checkpoints/ top level so benchmark/ never picks it up; deleted when training finishes).
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
    p.add_argument("--ckpt-dir", default="/content/drive/MyDrive/pruning-resnet18/checkpoints")
    p.add_argument("--results", default=os.path.join(REPO_ROOT, "results", "accuracy.csv"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--smoke", action="store_true", help="2 epochs on fake data, no download")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 2
    set_seed(args.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.ckpt_dir, exist_ok=True)
    final_path = os.path.join(args.ckpt_dir, "resnet18_baseline.pt")
    resume_dir = os.path.join(args.ckpt_dir, "_resume")
    os.makedirs(resume_dir, exist_ok=True)
    last_path = os.path.join(resume_dir, "baseline_last.pt")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size,
                                            args.num_workers, smoke=args.smoke, seed=args.seed)
    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_scaler(device)

    start_epoch = 0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M epochs={args.epochs} seed={args.seed}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch+1:3d}/{args.epochs} loss={loss:.4f} train={train_acc:.2f}% "
              f"test={test_acc:.2f}% lr={scheduler.get_last_lr()[0]:.4f} {time.time()-t0:.0f}s",
              flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch}, last_path)

    # Use the FINAL epoch (not best-on-test) to avoid tuning on the test set.
    torch.save(model.state_dict(), final_path)
    acc = evaluate(model, test_loader, device)
    print(f"\nBaseline accuracy: {acc:.2f}%  ->  saved {final_path}")
    upsert_row(args.results, {"method": "baseline", "sparsity": 0.0,
                              "acc_before_finetune": f"{acc:.2f}",
                              "acc_after_finetune": f"{acc:.2f}", "seed": args.seed})
    print(f"Wrote baseline row to {args.results}")
    os.remove(last_path)  # training finished; only resnet18_baseline.pt stays in checkpoints/


if __name__ == "__main__":
    main()
