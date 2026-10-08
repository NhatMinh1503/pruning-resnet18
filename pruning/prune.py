"""Magnitude pruning for ResNet-18 (CIFAR-10).

Owner: Hoa.  Place this file at pruning/prune.py in the repo.

Public API
----------
local_prune(model, sparsity)   -> pruned copy, each layer pruned to `sparsity`
global_prune(model, sparsity)  -> pruned copy, whole network pruned to `sparsity`
count_sparsity(model)          -> fraction of zero weights in Conv2d/Linear layers
make_permanent(model)          -> bake the masks into the weights (call AFTER fine-tuning)

Important about masks
---------------------
After local_prune / global_prune the model still holds a mask on every pruned
layer (weight = weight_orig * weight_mask). Keep it that way while you
fine-tune, so pruned weights stay at 0. Only call make_permanent() when
fine-tuning is finished and you are about to save the final checkpoint.
If you remove the mask before fine-tuning, the optimizer can regrow the
pruned weights and the sparsity is lost.

Only the `weight` of Conv2d and Linear layers is pruned. Biases and
BatchNorm parameters are left untouched.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def _prunable_layers(model):
    """Return [(module, 'weight'), ...] for every Conv2d and Linear layer."""
    return [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]


def _check_sparsity(sparsity):
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")


def local_prune(model, sparsity):
    """Layer-wise L1 magnitude pruning: every layer loses `sparsity` of its weights.

    Returns a pruned deep copy; the input model is not modified.
    """
    _check_sparsity(sparsity)
    pruned = copy.deepcopy(model)
    for module, name in _prunable_layers(pruned):
        prune.l1_unstructured(module, name=name, amount=sparsity)
    return pruned


def global_prune(model, sparsity):
    """Global L1 magnitude pruning: the smallest `sparsity` of ALL weights are removed.

    Layers with many small weights are pruned more than the others.
    Returns a pruned deep copy; the input model is not modified.
    """
    _check_sparsity(sparsity)
    pruned = copy.deepcopy(model)
    prune.global_unstructured(
        _prunable_layers(pruned),
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    return pruned


def count_sparsity(model):
    """Fraction of weights that are exactly 0 in Conv2d/Linear layers (0.0 to 1.0)."""
    zeros, total = 0, 0
    for module, name in _prunable_layers(model):
        w = getattr(module, name)  # masked weight if pruning is active
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return zeros / total if total else 0.0


def layer_sparsity(model):
    """Per-layer sparsity as a list of (layer_name, sparsity). Useful for Local vs Global plots."""
    result = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            result.append((name, float((w == 0).sum().item()) / w.numel()))
    return result


def make_permanent(model):
    """Remove pruning re-parametrization so weights are plain tensors with zeros.

    Call after fine-tuning, before saving the final checkpoint. Works in place
    and returns the model for convenience.
    """
    for module, name in _prunable_layers(model):
        if prune.is_pruned(module):
            try:
                prune.remove(module, name)
            except ValueError:
                pass  # this layer had no active pruning
    return model


if __name__ == "__main__":
    # Quick self-test on CPU with an untrained ResNet-18 (10 classes for CIFAR-10).
    from torchvision.models import resnet18

    torch.manual_seed(42)
    base = resnet18(num_classes=10)
    print(f"baseline sparsity: {count_sparsity(base):.4f}")

    for target in (0.1, 0.5, 0.9):
        loc = local_prune(base, target)
        glo = global_prune(base, target)
        print(
            f"target {target:.1f} | local {count_sparsity(loc):.4f} | "
            f"global {count_sparsity(glo):.4f}"
        )
        assert abs(count_sparsity(loc) - target) < 0.01
        assert abs(count_sparsity(glo) - target) < 0.01

    # The original model must be untouched.
    assert count_sparsity(base) < 1e-6

    # make_permanent keeps the zeros and the model still runs.
    glo = global_prune(base, 0.5)
    before = count_sparsity(glo)
    make_permanent(glo)
    assert abs(count_sparsity(glo) - before) < 1e-9
    out = glo(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    print("all checks passed")
