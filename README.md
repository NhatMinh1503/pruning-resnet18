# pruning-resnet18

AI Zemi 2026 H2 research: Magnitude Pruning on ResNet-18 with CIFAR-10 for model compression (AI Efficiency).

## Research questions

Q1. How does accuracy change as sparsity goes from 10% to 90%?

Q2. Local pruning (per layer) vs Global pruning (whole network): which keeps accuracy better?

Q3. How much accuracy does Train, Prune, Fine-tune recover?

Q4. Do FLOPs and model size reductions translate to real latency gains on actual hardware?

Q5. Optional: compare with structured (channel) pruning.

## Repository layout

pruning/ holds the pruning algorithms (local_prune and global_prune).

training/ holds baseline training, fine-tuning and accuracy evaluation.

benchmark/ holds FLOPs, model size and latency measurement.

results/ holds small CSV outputs only: accuracy.csv and efficiency.csv.

## Team workflow

Code is written in VS Code as .py files, pushed to GitHub, then pulled and run on Google Colab. Heavy files (checkpoints, datasets) live on the shared Google Drive and are never committed to Git. Colab is only a place to run code, not to store it.

Rule 1: always git pull before starting work.

Rule 2: work on your own branch, open a Pull Request, and have a teammate review it before merging to main.

Rule 3: only edit your own folder to avoid conflicts.

Rule 4: after finishing a step, post one line in the team chat saying what is done and where the files are.

## Hand-off order

Step 1: Khang trains the baseline ResNet-18 and saves resnet18_baseline.pt to Drive.

Step 2: Hoa implements pruning/prune.py with local_prune and global_prune.

Step 3: Khang runs Prune then Fine-tune for every sparsity, saves checkpoints to Drive and writes results/accuracy.csv.

Step 4: Canh runs benchmark/ on every checkpoint and writes results/efficiency.csv.

Step 5: everyone merges the CSVs, plots, and writes the report together.

## Checkpoint naming

resnet18_METHOD_SPARSITY_STAGE.pt where METHOD is local or global, SPARSITY is like 0.5, and STAGE is raw or finetuned. Example: resnet18_global_0.5_finetuned.pt

## Setup

Run pip install -r requirements.txt
