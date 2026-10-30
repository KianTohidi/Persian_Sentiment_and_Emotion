"""Summarize the fine-tuning runs and select, for each model and task, the run with the best validation macro-F1."""
import pandas as pd

from common import FINE_TUNED, RESULTS, SEEDS, TASKS

log = pd.read_csv(RESULTS / "all_runs_log.csv")
expected = len(FINE_TUNED) * len(TASKS) * len(SEEDS)
print(f"runs in log: {len(log)} (expected {expected}); duplicate run names: {log['run'].duplicated().sum()}")

summary = log.groupby(["task", "model"]).agg(
    runs=("seed", "count"),
    acc_mean=("test_acc", "mean"), acc_sd=("test_acc", "std"),
    f1_mean=("test_f1", "mean"), f1_sd=("test_f1", "std"),
    train_min_mean=("train_min", "mean"), train_min_min=("train_min", "min"), train_min_max=("train_min", "max"),
    infer_sec_mean=("infer_sec", "mean"))
percent_columns = ["acc_mean", "acc_sd", "f1_mean", "f1_sd"]
summary[percent_columns] = summary[percent_columns] * 100
summary = summary.round(2)
summary.to_csv(RESULTS / "baselines_summary.csv")

best = log.loc[log.groupby(["task", "model"])["val_f1"].idxmax(),
               ["task", "model", "seed", "run", "val_f1", "test_acc", "test_f1"]]
best.to_csv(RESULTS / "chosen_runs.csv", index=False)

print(summary.to_string())
print(best.to_string(index=False))
