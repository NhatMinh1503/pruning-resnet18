# training/

Baseline training, Train -> Prune -> Fine-tune, and accuracy evaluation. Owner: Khang.

Outputs: checkpoints go to Google Drive (not Git); accuracy results go to results/accuracy.csv.

## Files

| File | Purpose |
|---|---|
| `common.py` | `build_model()` (CIFAR ResNet-18), data loaders, `train_one_epoch`, `evaluate`, seed=42 |
| `train.py` | Step 1: train baseline -> `resnet18_baseline.pt` |
| `run_pruning.py` | Step 3: local/global x 10%..90% -> prune, eval, fine-tune, eval -> `accuracy.csv` |

**Loading any checkpoint (benchmark/ too):**

```python
from training.common import build_model, load_weights
model = load_weights(build_model(), "checkpoints/resnet18_global_0.5_finetuned.pt")
```

All saved `.pt` files are plain `state_dict`s (no `weight_orig`/`weight_mask`).

## Settings

| | Baseline | Fine-tune |
|---|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov, wd 5e-4 | same |
| LR | 0.1, cosine | 0.01, cosine |
| Epochs | 100 | 10 per (method, sparsity) |
| Batch | 128 | 128 |
| Augment | RandomCrop(32, pad 4) + HFlip | same |
| Seed | 42 | 42 |

Fine-tuning re-applies the pruning mask after every optimizer step, so pruned weights stay 0.

## Run on Colab

**Easiest:** open [`training/colab_training.ipynb`](colab_training.ipynb) in Colab
([open directly](https://colab.research.google.com/github/NhatMinh1503/pruning-resnet18/blob/main/training/colab_training.ipynb))
and run the cells top to bottom. GPU (T4) is preset.

One-time setup: Google Drive → *Shared with me* → right-click `pruning-project` →
**Organize → Add shortcut → My Drive**, so Colab can see it at `/content/drive/MyDrive/pruning-project`.

| Shared Drive path | Content |
|---|---|
| `pruning-project/checkpoints/` | `resnet18_baseline.pt` + `resnet18_{method}_{sparsity}_{raw\|finetuned}.pt` |
| `pruning-project/results/accuracy.csv` | Latest accuracy table (copied back to `results/` in Git) |

Manual commands (same as the notebook):

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/NhatMinh1503/pruning-resnet18.git
%cd pruning-resnet18
CK = "/content/drive/MyDrive/pruning-project/checkpoints"
!python training/train.py --data-dir /content/data --ckpt-dir {CK}          # baseline (done 2026-10-12: 95.10%)
!python training/run_pruning.py --data-dir /content/data --ckpt-dir {CK} --methods local
!python training/run_pruning.py --data-dir /content/data --ckpt-dir {CK} --methods global
```

Measured on Colab T4: ~28 s/epoch (baseline 100 epochs ≈ 47 min). Scripts resume / skip finished work after a disconnect.

Pipeline test without GPU / dataset: `python training/train.py --smoke --ckpt-dir ./ck`.
