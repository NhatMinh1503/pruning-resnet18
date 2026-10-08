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

## Run on Colab (Runtime -> GPU)

```python
from google.colab import drive; drive.mount('/content/drive')
# branch `training` must be pushed to GitHub first (git push -u origin training)
!git clone -b training https://github.com/NhatMinh1503/pruning-resnet18.git
%cd pruning-resnet18
!pip install -q thop fvcore

CK = "/content/drive/MyDrive/pruning-resnet18/checkpoints"
# Step 1 (resumes automatically if Colab disconnects)
!python training/train.py --ckpt-dir $CK
# Step 3 (after pruning/prune.py is merged); can be split by --methods / --sparsities
!python training/run_pruning.py --ckpt-dir $CK --methods local
!python training/run_pruning.py --ckpt-dir $CK --methods global
```

Pipeline test without GPU / dataset: `python training/train.py --smoke --ckpt-dir ./ck`.
