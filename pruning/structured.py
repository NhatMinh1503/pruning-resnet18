"""Structured (channel) pruning for the CIFAR ResNet-18 (research question Q5).

Owner: Hoa.

Unlike magnitude pruning in prune.py (which only multiplies weights by a 0/1
mask, so tensors stay dense), this file REALLY removes channels: the pruned
model has smaller Conv2d / BatchNorm layers, so FLOPs, file size and latency
all go down on ordinary hardware.

What is pruned
--------------
Only the channel between the two convs inside every BasicBlock
(conv1 output = bn1 = conv2 input). Those channels do not touch the residual
connection, so no other layer has to change. The stem, the downsample convs
and the block outputs stay as they are.

Channels are ranked by the L1 norm of their conv1 filter (same idea as
magnitude pruning, but per filter instead of per weight; Li et al. 2017,
"Pruning Filters for Efficient ConvNets").

Public API
----------
channel_prune(model, ratio)   -> NEW smaller model, `ratio` of the inner channels removed
block_widths(model)           -> inner width of every BasicBlock, e.g. [64, 64, 128, ...]
build_structured(widths)      -> empty model with those widths (to load a checkpoint)
save_structured(model, path)  /  load_structured(path)

A structured model is NOT compatible with build_model(): always save it with
save_structured() (it stores the widths together with the weights).

Self-test:  python pruning/structured.py
"""
import copy
import os
import sys

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from training.common import build_model  # noqa: E402


def _blocks(model):
    return [m for m in model.modules() if isinstance(m, BasicBlock)]


def block_widths(model):
    """Inner channel count (conv1 out channels) of every BasicBlock, in order."""
    return [b.conv1.out_channels for b in _blocks(model)]


def _shrink_block(block, keep):
    """Keep only the inner channels listed in `keep` (LongTensor of indices)."""
    c1, bn, c2 = block.conv1, block.bn1, block.conv2
    k = len(keep)

    new_c1 = nn.Conv2d(c1.in_channels, k, c1.kernel_size, c1.stride, c1.padding, bias=False)
    new_c1.weight.data = c1.weight.data[keep].clone()

    new_bn = nn.BatchNorm2d(k, eps=bn.eps, momentum=bn.momentum)
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(new_bn, name).data = getattr(bn, name).data[keep].clone()
    new_bn.num_batches_tracked = bn.num_batches_tracked.clone()

    new_c2 = nn.Conv2d(k, c2.out_channels, c2.kernel_size, c2.stride, c2.padding, bias=False)
    new_c2.weight.data = c2.weight.data[:, keep].clone()

    block.conv1, block.bn1, block.conv2 = new_c1, new_bn, new_c2


def channel_prune(model, ratio):
    """Remove `ratio` of the inner channels of every BasicBlock (L1-norm ranking).

    Returns a new, physically smaller model; the input model is not modified.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    pruned = copy.deepcopy(model)
    for block in _blocks(pruned):
        n = block.conv1.out_channels
        k = max(1, int(round(n * (1.0 - ratio))))
        score = block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))  # L1 norm per filter
        keep = torch.argsort(score, descending=True)[:k].sort().values
        _shrink_block(block, keep)
    return pruned


def build_structured(widths):
    """CIFAR ResNet-18 whose BasicBlocks have the given inner widths (random weights)."""
    model = build_model()
    blocks = _blocks(model)
    if len(widths) != len(blocks):
        raise ValueError(f"expected {len(blocks)} widths, got {len(widths)}")
    for block, k in zip(blocks, widths):
        _shrink_block(block, torch.arange(k))
    return model


def save_structured(model, path):
    torch.save({"widths": block_widths(model), "model": model.state_dict()}, path)


def load_structured(path, device="cpu"):
    obj = torch.load(path, map_location=device)
    model = build_structured(obj["widths"])
    model.load_state_dict(obj["model"])
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(42)
    base = build_model().eval()
    x = torch.randn(2, 3, 32, 32)
    n0 = count_params(base)
    assert block_widths(base) == [64, 64, 128, 128, 256, 256, 512, 512]

    # ratio 0 keeps the network identical
    same = channel_prune(base, 0.0).eval()
    assert torch.allclose(same(x), base(x), atol=1e-5)

    for r in (0.25, 0.5, 0.75):
        m = channel_prune(base, r).eval()
        assert m(x).shape == (2, 10)
        print(f"ratio {r:.2f}: widths {block_widths(m)}  params {count_params(m)/1e6:.2f}M "
              f"({100*count_params(m)/n0:.0f}% of baseline)")

    # the input model is untouched
    assert block_widths(base) == [64, 64, 128, 128, 256, 256, 512, 512]

    # removing the channels with the smallest L1 norm keeps them out of the new model:
    # a channel whose conv1 filter is all zero must be the first one dropped
    z = copy.deepcopy(base)
    z.layer1[0].conv1.weight.data[5] = 0
    m = channel_prune(z, 1 / 64)
    kept = m.layer1[0].conv1.weight.abs().sum(dim=(1, 2, 3))
    assert m.layer1[0].conv1.out_channels == 63 and bool((kept > 0).all())

    # save / load round trip
    m = channel_prune(base, 0.5).eval()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.pt")
        save_structured(m, p)
        m2 = load_structured(p).eval()
        assert torch.allclose(m(x), m2(x), atol=1e-6)
    print("all checks passed")
