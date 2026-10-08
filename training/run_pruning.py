"""Step 3: Prune -> evaluate -> Fine-tune -> evaluate, for every method x sparsity.

Needs pruning/prune.py (Hoa) with:
    local_prune(model, sparsity) -> model
    global_prune(model, sparsity) -> model

Saves to --ckpt-dir:
    resnet18_{local|global}_{sparsity}_{raw|finetuned}.pt   (plain state_dict)
Writes results/accuracy.csv:
    method, sparsity, acc_before_finetune, acc_after_finetune, seed

Resumable: (method, sparsity) pairs already in the CSV with a finetuned
checkpoint on Drive are skipped. Split long runs, e.g.
    python training/run_pruning.py --methods local
    python training/run_pruning.py --methods global --sparsities 0.7,0.8,0.9
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from training.common import (SEED, build_model, evaluate, get_device, get_loaders,  # noqa: E402
                             load_weights, make_scaler, read_rows, set_seed,
                             train_one_epoch, upsert_row)

try:
    from pruning.prune import global_prune, local_prune  # noqa: E402
except ImportError as e:
    sys.exit(f"Cannot import pruning/prune.py ({e}). Pull Hoa's branch first.")

PRUNE_FNS = {"local": local_prune, "global": global_prune}
DEFAULT_SPARSITIES = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--ckpt-dir", default="/content/drive/MyDrive/pruning-resnet18/checkpoints")
    p.add_argument("--results", default=os.path.join(REPO_ROOT, "results", "accuracy.csv"))
    p.add_argument("--methods", default="local,global")
    p.add_argument("--sparsities", default=DEFAULT_SPARSITIES)
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--ft-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true", help="re-run pairs already done")
    p.add_argument("--smoke", action="store_true", help="1 fine-tune epoch on fake data")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Mask helpers: work whether prune.py uses torch.nn.utils.prune or zeroes
# weights manually.
# --------------------------------------------------------------------------- #
def make_permanent(model: nn.Module) -> None:
    """Remove torch.nn.utils.prune reparametrization (weight_orig/weight_mask)
    so the state_dict has plain keys that build_model() can load."""
    for module in model.modules():
        for name, _ in list(module.named_parameters(recurse=False)):
            if name.endswith("_orig"):
                torch_prune.remove(module, name[:-5])


def collect_masks(model: nn.Module) -> dict:
    """Binary masks (1 = kept) for every Conv/Linear weight."""
    return {name: (m.weight != 0).float()
            for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def apply_masks(model: nn.Module, masks: dict) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, mask in masks.items():
            modules[name].weight.mul_(mask)


def measured_sparsity(model: nn.Module) -> float:
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += (m.weight == 0).sum().item()
            total += m.weight.numel()
    return zeros / total


def ckpt_name(method: str, sparsity: float, stage: str) -> str:
    return f"resnet18_{method}_{sparsity:g}_{stage}.pt"


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.smoke:
        args.ft_epochs = 1
    device = get_device()
    torch.backends.cudnn.benchmark = True
    methods = [m.strip() for m in args.methods.split(",")]
    sparsities = [round(float(s), 2) for s in args.sparsities.split(",")]
    for m in methods:
        if m not in PRUNE_FNS:
            sys.exit(f"Unknown method {m!r}, use local/global")

    baseline_path = os.path.join(args.ckpt_dir, "resnet18_baseline.pt")
    if not os.path.exists(baseline_path):
        sys.exit(f"Missing {baseline_path}. Run training/train.py first.")
    baseline_state = torch.load(baseline_path, map_location="cpu")

    set_seed(args.seed)
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size,
                                            args.num_workers, smoke=args.smoke, seed=args.seed)

    base = build_model().to(device)
    base.load_state_dict(baseline_state)
    print(f"Baseline accuracy: {evaluate(base, test_loader, device):.2f}%\n")
    del base

    done = {(r["method"], r["sparsity"]) for r in read_rows(args.results)}

    for method in methods:
        for s in sparsities:
            ft_path = os.path.join(args.ckpt_dir, ckpt_name(method, s, "finetuned"))
            if not args.force and (method, str(s)) in done and os.path.exists(ft_path):
                print(f"[skip] {method} {s} already done")
                continue
            t0 = time.time()
            set_seed(args.seed)  # same randomness for every run

            # 1) Prune a fresh copy of the baseline
            model = build_model()
            model.load_state_dict(baseline_state)
            model = PRUNE_FNS[method](model, s) or model  # in case prune fn edits in place
            make_permanent(model)
            model.to(device)
            real_s = measured_sparsity(model)
            acc_raw = evaluate(model, test_loader, device)
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, ckpt_name(method, s, "raw")))

            # 2) Fine-tune, keeping pruned weights at zero
            masks = {k: v.to(device) for k, v in collect_masks(model).items()}
            optimizer = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9,
                                        weight_decay=5e-4, nesterov=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs)
            scaler = make_scaler(device)
            for ep in range(args.ft_epochs):
                loss, _ = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                          after_step=lambda: apply_masks(model, masks))
                scheduler.step()
                print(f"  {method} {s} ft {ep+1}/{args.ft_epochs} loss={loss:.4f}", flush=True)
            acc_ft = evaluate(model, test_loader, device)
            assert abs(measured_sparsity(model) - real_s) < 1e-6, "pruned weights came back!"
            torch.save(model.state_dict(), ft_path)

            upsert_row(args.results, {"method": method, "sparsity": s,
                                      "acc_before_finetune": f"{acc_raw:.2f}",
                                      "acc_after_finetune": f"{acc_ft:.2f}", "seed": args.seed})
            print(f"[done] {method:6s} target={s:.1f} real={real_s:.3f} "
                  f"raw={acc_raw:.2f}% finetuned={acc_ft:.2f}% ({time.time()-t0:.0f}s)\n")

    print(f"Results: {args.results}")


if __name__ == "__main__":
    main()
