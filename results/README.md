# results/

Experiment outputs (small CSV files only).

- accuracy.csv: method, sparsity, acc_before_finetune, acc_after_finetune, seed
- efficiency.csv: method, sparsity, flops, size_mb, latency_ms
- bn_recalib.csv: method, sparsity, real_sparsity, acc_raw, acc_bn_recalib, n_calib, seed
  (BN statistics re-estimated on 2000 augmented train images, no weight update; `pruning/bn_recalib.py`)
- bn_recalib_clean.csv / bn_recalib_clean_ncalib.csv: same with clean (non-augmented) calibration
  images, and accuracy vs number of calibration images (ablation)
- bn_aware.csv: method, criterion, schedule, sparsity, real_sparsity, acc_raw, acc_bn_recalib, n_calib, seed
  (ranking by |w| vs by BN-folded weight |w|*|gamma|/sigma, one-shot vs 5-step iterative,
  BN re-calibrated, no fine-tune; `pruning/bn_aware.py`)
- bn_aware_stages.csv: share of pruned weights per stage (conv1, layer1-4, fc) for each row above
