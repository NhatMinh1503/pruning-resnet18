# pruning/

Magnitude Pruning (L1, unstructured) for ResNet-18 / CIFAR-10. Owner: Hoa.

Only the `weight` of Conv2d and Linear layers is pruned (21 layers in the CIFAR ResNet-18).
Biases and BatchNorm parameters are left untouched.

## Files

| File | Purpose |
|---|---|
| `prune.py` | Pruning API used by `training/run_pruning.py` |
| `analyze_layers.py` | Per-layer sparsity of Local vs Global pruning (Q2) -> CSV + figure |

## API (`prune.py`)

| Function | Returns | Notes |
|---|---|---|
| `local_prune(model, sparsity)` | pruned copy | every layer loses `sparsity` of its own weights |
| `global_prune(model, sparsity)` | pruned copy | the smallest `sparsity` of all weights in the network are removed, so layers end up with different sparsity |
| `count_sparsity(model)` | float, 0.0 to 1.0 | fraction of zero weights over all prunable layers |
| `layer_sparsity(model)` | list of `(layer_name, sparsity)` | per-layer breakdown |
| `make_permanent(model)` | the same model | removes the masks and bakes the zeros into `weight` (in place) |

`sparsity` must be in `[0, 1)`. The input model is never modified by `local_prune` / `global_prune`.

```python
from training.common import build_model, load_weights
from pruning.prune import global_prune, count_sparsity, layer_sparsity, make_permanent

base = load_weights(build_model(), "resnet18_baseline.pt")
model = global_prune(base, 0.5)
print(count_sparsity(model))      # 0.5
print(layer_sparsity(model)[:3])  # [('conv1', ...), ('layer1.0.conv1', ...), ...]
```

## Masks and fine-tuning

After pruning, each layer holds `weight_orig` and `weight_mask`, and the model uses
`weight = weight_orig * weight_mask`. Keep the mask on while fine-tuning so pruned
weights stay at 0. If it is removed too early, the optimizer regrows them.

Order: prune -> fine-tune (mask on) -> `make_permanent` -> save checkpoint.

## Per-layer analysis (`analyze_layers.py`)

Runs on CPU in a few seconds; no GPU and no dataset needed.

```bash
# real analysis: download resnet18_baseline.pt from the team Drive first
python pruning/analyze_layers.py --checkpoint path/to/resnet18_baseline.pt

# pipeline test with an untrained model (numbers are not meaningful)
python pruning/analyze_layers.py --random-init
```

Options: `--sparsities 0.3,0.5,0.7,0.9` (default), `--csv`, `--fig`.

Outputs:

- `results/layer_sparsity.csv`: `method, sparsity, layer, num_params, layer_sparsity`
- `results/figures/layer_sparsity.png`: global per-layer sparsity (local shown as dashed lines at the target) and the number of weights per layer

## Self-test

```bash
python pruning/prune.py   # prints "all checks passed"
```

## Experiments (Q2, Q5)

| File | Output |
|---|---|
| `ablate_layers.py` | `results/ablation_layers.csv`: keep fc / conv1 / downsample dense |
| `ablate_stages.py` | `results/ablation_stages.csv`: prune only layer1..4, matched sparsity |
| `structured.py`, `run_structured.py` | `results/structured.csv` (Colab, with fine-tune), `results/structured_raw.csv` (no fine-tune) |
| `plot_tradeoff.py`, `plot_ablation.py` | `results/figures/q2_q5_tradeoff.png`, `results/figures/q2_stage_ablation.png` |
| `colab_pruning.ipynb` | runs all of the above on Colab |

Results so far (test accuracy right after pruning, no fine-tune, baseline 95.10%):

- Local collapses at 90% because of **layer2 and layer3**: pruning only layer3 to 90% (18.5% of the
  weights) gives 56.8%, pruning only layer2 gives 69.4%, while pruning only layer4 (74% of the
  weights) still gives 93.8%. Global keeps layer2/3 at ~56-83% sparsity, which is why it survives.
- Small layers: keeping fc + conv1 + downsample dense helps at 80% (63.4% -> 72.0% at the same
  total sparsity) but not at 90%.
- Structured (channel) pruning before fine-tune: 25% -> 92.6%, 50% -> 67.2%, 75% -> 14.2%, so it
  needs fine-tuning much more than unstructured pruning (Local 50% is still 94.4%).
