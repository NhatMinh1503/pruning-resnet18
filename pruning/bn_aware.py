"""BN-aware magnitude pruning, one-shot vs iterative with BN re-calibration
(no fine-tune).

Owner: Hoa.

Why this experiment
-------------------
`bn_recalib.py` showed that most of the accuracy collapse right after pruning
is BatchNorm being out of date, not lost information. The obvious follow-up
("prune a little -> recalibrate BN -> prune a little more ...") does NOTHING
for plain magnitude pruning: re-calibration only changes BN buffers, never
the weights, so the ranking by |w| is the same at every step and the final
mask is identical to one-shot pruning. (Same for structured L1 ranking.)
`--check-magnitude-iter` verifies this.

What does change with BN is the *effective* weight. In ResNet every conv is
followed by a BN, and at inference the pair computes

    y = gamma * (conv(x, w) - mean) / sqrt(var + eps) + beta
      = conv(x, w * gamma / sqrt(var + eps)) + const

so the weight the network really uses is w_eff = w * |gamma| / sigma (per
output channel). Ranking by |w| ignores that BN can blow up or shrink every
channel, which matters most for GLOBAL pruning, where |w| of different layers
are compared with each other.

Criteria
--------
magnitude  : score = |w|                         (prune.py, the baseline)
bn_folded  : score = |w| * |gamma| / sigma       (BN of the following layer)
             structured: L1(conv1 filter) * |gamma| / sigma of bn1
             fc (no BN after it) keeps score = |w|.

Schedules
---------
oneshot : prune to the target in one step, then re-calibrate BN once.
iter{K} : K steps, sparsity s_k = 1 - (1 - s)^(k/K) (same fraction of the
          remaining weights each step). After every step BN is re-calibrated,
          so sigma - and therefore the bn_folded scores - are up to date for
          the next step. Already-pruned weights have score 0 and stay pruned.

Every row is evaluated after the final BN re-calibration (2000 augmented
train images, same subset as bn_recalib.py). Weights are never trained.

Outputs
-------
results/bn_aware.csv
    method, criterion, schedule, sparsity, real_sparsity, acc_raw,
    acc_bn_recalib, n_calib, seed
    (acc_raw = before any re-calibration; only for one-shot rows)
results/bn_aware_stages.csv
    method, criterion, schedule, sparsity, stage, stage_sparsity
    (where the pruning goes: conv1 / layer1..4 / fc, unstructured only)

Result (seed 42, see results/bn_aware.csv)
-----------------------------------------
BN-folded ranking is worse than |w| in every case but one (Local 0.95
iterative, 31.2 vs 28.8%, both unusable), worst for Global (0.9: 92.1 -> 45.8%
one-shot, 83.8% iterative). gamma/sigma is ~3.9x in layer4 but ~1.8x in
layer2, so folding makes layer4 look important and moves the zeros into the
sensitive layer2/layer3 (Global 0.9: layer2 62% -> 80% pruned). Iterative
re-calibration helps the folded ranking (+0.3 to +38 pts) but never catches up
with plain |w|. Plain |w| works for Global *because* the small raw weights of
layer4 attract most of the pruning.

Usage
-----
    python pruning/bn_aware.py --checkpoint {CK}/resnet18_baseline.pt --data-dir /content/data
    python pruning/bn_aware.py --random-init --smoke     # pipeline test, fake data
"""
import argparse
import copy
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as tprune
from torchvision.models.resnet import BasicBlock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from pruning.bn_recalib import calib_loader, get_datasets, recalibrate_bn, write_csv  # noqa: E402
from pruning.prune import _prunable_layers, count_sparsity, make_permanent  # noqa: E402
from pruning.structured import _blocks, _shrink_block  # noqa: E402
from training.common import SEED, build_model, evaluate, get_device, load_weights, set_seed  # noqa: E402

CSV_HEADER = ["method", "criterion", "schedule", "sparsity", "real_sparsity", "acc_raw",
              "acc_bn_recalib", "n_calib", "seed"]
STAGE_HEADER = ["method", "criterion", "schedule", "sparsity", "stage", "stage_sparsity"]
STAGES = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def conv_bn_pairs(model):
    """{conv module: BN module that normalizes its output} for a torchvision ResNet."""
    pairs = {model.conv1: model.bn1}
    for b in _blocks(model):
        pairs[b.conv1] = b.bn1
        pairs[b.conv2] = b.bn2
        if b.downsample is not None:
            pairs[b.downsample[0]] = b.downsample[1]
    return pairs


def bn_scale(bn):
    """|gamma| / sqrt(running_var + eps): how much BN multiplies each channel."""
    return (bn.weight.detach().abs() / torch.sqrt(bn.running_var + bn.eps)).detach()


@torch.no_grad()
def scores(model, criterion):
    """{module: importance tensor (same shape as weight)} for every Conv2d/Linear."""
    pairs = conv_bn_pairs(model) if criterion == "bn_folded" else {}
    out = {}
    for m, _ in _prunable_layers(model):
        s = m.weight.detach().abs()
        if m in pairs:
            s = s * bn_scale(pairs[m]).view(-1, *([1] * (s.dim() - 1)))
        out[m] = s
    return out


def prune_unstructured(model, method, sparsity, criterion):
    """In place: prune `model` to `sparsity` with the given ranking, then bake the zeros.

    Weights that are already 0 have score 0, so they stay in the pruned set
    (this is what makes repeated calls an iterative schedule)."""
    sc = scores(model, criterion)
    params = _prunable_layers(model)
    if method == "global":
        tprune.global_unstructured(params, pruning_method=tprune.L1Unstructured,
                                   importance_scores={(m, n): sc[m] for m, n in params},
                                   amount=sparsity)
    else:
        for m, n in params:
            tprune.l1_unstructured(m, n, amount=sparsity, importance_scores=sc[m])
    return make_permanent(model)


def prune_structured(model, orig_widths, ratio, criterion):
    """In place: every BasicBlock keeps round(orig_width * (1 - ratio)) inner channels."""
    for block, n in zip(_blocks(model), orig_widths):
        k = max(1, int(round(n * (1.0 - ratio))))
        if block.conv1.out_channels <= k:
            continue
        s = block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))
        if criterion == "bn_folded":
            s = s * bn_scale(block.bn1)
        keep = torch.argsort(s, descending=True)[:k].sort().values
        _shrink_block(block, keep)
    return model


def schedule(target, steps):
    """[s_1, ..., s_K]: same fraction of the remaining weights removed at every step."""
    return [1.0 - (1.0 - target) ** (k / steps) for k in range(1, steps + 1)]


# --------------------------------------------------------------------------- #
def stage_sparsity(model):
    zeros, total = {}, {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            st = name.split(".")[0]
            zeros[st] = zeros.get(st, 0) + int((m.weight == 0).sum())
            total[st] = total.get(st, 0) + m.weight.numel()
    return {st: zeros[st] / total[st] for st in STAGES if st in total}


def masks_equal(a, b):
    la, lb = _prunable_layers(a), _prunable_layers(b)
    return all(torch.equal(ma.weight == 0, mb.weight == 0) for (ma, _), (mb, _) in zip(la, lb))


def run(base, method, criterion, steps, target, cal, device):
    """Returns (pruned + BN-recalibrated model, model before any recalibration or None)."""
    model = copy.deepcopy(base).to(device)
    widths = [b.conv1.out_channels for b in _blocks(model)]
    raw = None
    for i, s in enumerate(schedule(target, steps)):
        if method == "structured":
            prune_structured(model, widths, s, criterion)
        else:
            prune_unstructured(model, method, s, criterion)
        if steps == 1:
            raw = copy.deepcopy(model)
        recalibrate_bn(model, cal, device)
    return model, raw


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to resnet18_baseline.pt")
    src.add_argument("--random-init", action="store_true", help="untrained model (pipeline test)")
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--methods", default="global,local,structured")
    p.add_argument("--sparsities", default="0.8,0.9,0.95")
    p.add_argument("--ratios", default="0.5,0.75", help="structured channel ratios")
    p.add_argument("--steps", type=int, default=5, help="K of the iterative schedule")
    p.add_argument("--n-calib", type=int, default=2000)
    p.add_argument("--check-magnitude-iter", action="store_true",
                   help="also run magnitude + iterative and check the mask equals one-shot")
    p.add_argument("--csv", default=os.path.join(REPO_ROOT, "results", "bn_aware.csv"))
    p.add_argument("--smoke", action="store_true", help="fake data")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 1)
    device = get_device()
    base = build_model()
    if args.checkpoint:
        load_weights(base, args.checkpoint)
    base.eval()

    train_set, test_set = get_datasets(args.data_dir, args.smoke, aug=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    n_calib = min(args.n_calib, len(train_set))
    cal = calib_loader(train_set, n_calib)

    rows, stage_rows = [], []
    methods = args.methods.split(",")
    for method in methods:
        targets = args.ratios if method == "structured" else args.sparsities
        for target in [round(float(t), 2) for t in targets.split(",")]:
            configs = [("magnitude", 1), ("bn_folded", 1), ("bn_folded", args.steps)]
            if args.check_magnitude_iter and method != "structured":
                configs.append(("magnitude", args.steps))
            ref = {}
            for criterion, steps in configs:
                t0 = time.time()
                set_seed(SEED)  # same augmentation draws for every configuration
                model, raw = run(base, method, criterion, steps, target, cal, device)
                sched = "oneshot" if steps == 1 else f"iter{steps}"
                a_raw = evaluate(raw, test_loader, device) if raw is not None else None
                a_cal = evaluate(model, test_loader, device)
                rs = target if method == "structured" else count_sparsity(model)
                note = ""
                if method != "structured":
                    ref[(criterion, steps)] = model
                    if criterion == "magnitude" and steps > 1:
                        same = masks_equal(model, ref[("magnitude", 1)])
                        note = f"  mask == oneshot: {same}"
                    for st, v in stage_sparsity(model).items():
                        stage_rows.append({"method": method, "criterion": criterion,
                                           "schedule": sched, "sparsity": target,
                                           "stage": st, "stage_sparsity": f"{v:.4f}"})
                rows.append({"method": method, "criterion": criterion, "schedule": sched,
                             "sparsity": target, "real_sparsity": f"{rs:.4f}",
                             "acc_raw": "" if a_raw is None else f"{a_raw:.2f}",
                             "acc_bn_recalib": f"{a_cal:.2f}", "n_calib": n_calib, "seed": SEED})
                raw_txt = "" if a_raw is None else f"raw={a_raw:6.2f}%  "
                print(f"{method:10s} {target:.2f} {criterion:9s} {sched:7s} {raw_txt}"
                      f"bn_recalib={a_cal:6.2f}%  ({time.time() - t0:.0f}s){note}", flush=True)
            write_csv(args.csv, CSV_HEADER, rows)  # write as we go (long CPU runs)
            if stage_rows:
                write_csv(args.csv.replace(".csv", "_stages.csv"), STAGE_HEADER, stage_rows)


if __name__ == "__main__":
    main()
