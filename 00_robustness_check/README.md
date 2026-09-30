# Batch vs. Single-Instance Robustness Check

Code and raw data for the batch (20) vs. single-instance (1) comparison reported in Section 4.3.

- `analyze_robustness.py` — run on any file in `data/` to get accuracy, Cohen's kappa, and McNemar's test.
- `data/` — all 8 raw result files (one per model per task). Columns: text, ground truth, batch20 prediction, batch1 prediction.

Full methodology and results: Sections 3.4.6 and 4.3 of the paper.
