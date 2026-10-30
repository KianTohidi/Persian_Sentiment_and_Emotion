"""Compare the seven models on the common test sets.

Outputs (results/): final_scores_with_CI.csv (accuracy and macro-F1 with bootstrap 95% CIs),
pairwise_tests.csv (exact McNemar tests, Holm-corrected) and seed_robustness.csv.
Invalid LLM answers are counted as errors.
"""
import itertools

import numpy as np
import pandas as pd

from common import (FINE_TUNED, LABELS, LLMS, MODELS, RESULTS, SEEDS, TASKS, accuracy_macro_f1, encode,
                    holm, load_predictions, mcnemar_exact, seed_predictions)

N_BOOT = 2000
rng = np.random.default_rng(0)
score_rows, pair_rows, seed_rows = [], [], []

for task in TASKS:
    df = load_predictions(task)
    k = len(LABELS[task])
    y = encode(df["true"].values, task)
    pred = {m: encode(df[m].values, task) for m in MODELS}
    n = len(y)

    # Bootstrap: every model is re-scored on the same resampled texts.
    boot = {m: np.zeros((N_BOOT, 2)) for m in MODELS}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for m in MODELS:
            boot[m][b] = accuracy_macro_f1(y[idx], pred[m][idx], k)
    for m in MODELS:
        acc, f1 = accuracy_macro_f1(y, pred[m], k)
        acc_low, acc_high = np.percentile(boot[m][:, 0], [2.5, 97.5])
        f1_low, f1_high = np.percentile(boot[m][:, 1], [2.5, 97.5])
        score_rows.append({
            "task": task, "model": MODELS[m], "type": "LLM" if m in LLMS else "Fine-tuned",
            "acc": acc * 100, "acc_lo": acc_low * 100, "acc_hi": acc_high * 100,
            "f1": f1 * 100, "f1_lo": f1_low * 100, "f1_hi": f1_high * 100})

    # Pairwise exact McNemar tests with Holm correction (21 pairs per task).
    correct = {m: pred[m] == y for m in MODELS}
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        a_only, b_only, p = mcnemar_exact(correct[a], correct[b])
        rows.append({"task": task, "A": MODELS[a], "B": MODELS[b], "A_only_right": a_only,
                     "B_only_right": b_only, "diff_pp": (correct[a].mean() - correct[b].mean()) * 100,
                     "p_raw": p})
    for row, p_adjusted in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"] = p_adjusted
        row["real_difference"] = "YES" if p_adjusted < 0.05 else "no"
    pair_rows += rows

    # Robustness: each fine-tuned run (5 seeds) against each LLM, unadjusted p < 0.05.
    for ft in FINE_TUNED:
        for llm in LLMS:
            ahead = ft_better = llm_better = 0
            differences = []
            for seed in SEEDS:
                ft_correct = encode(seed_predictions(task, ft, seed, df), task) == y
                a_only, b_only, p = mcnemar_exact(ft_correct, correct[llm])
                differences.append((ft_correct.mean() - correct[llm].mean()) * 100)
                ahead += ft_correct.mean() > correct[llm].mean()
                if p < 0.05:
                    ft_better += a_only > b_only
                    llm_better += a_only <= b_only
            seed_rows.append({
                "task": task, "fine-tuned": FINE_TUNED[ft], "LLM": LLMS[llm],
                "seeds FT ahead": f"{ahead}/{len(SEEDS)}", "sig. FT better": f"{ft_better}/{len(SEEDS)}",
                "sig. LLM better": f"{llm_better}/{len(SEEDS)}",
                "mean diff (pp)": round(float(np.mean(differences)), 2)})

scores = pd.DataFrame(score_rows)
scores.to_csv(RESULTS / "final_scores_with_CI.csv", index=False)
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(RESULTS / "pairwise_tests.csv", index=False)
pd.DataFrame(seed_rows).to_csv(RESULTS / "seed_robustness.csv", index=False)

for task in TASKS:
    print(f"\n{task}: scores (evaluated run) and number of significant pairs")
    print(scores[scores["task"] == task].round(2).to_string(index=False))
    print("significant after Holm:", int((pairs[pairs["task"] == task]["real_difference"] == "YES").sum()), "of 21")
