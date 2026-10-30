"""Per-class F1, most frequent confusions and error overlap between the seven models."""
import numpy as np
import pandas as pd

from common import (FINE_TUNED, LABELS, LLMS, MODELS, RESULTS, SEEDS, TASKS, confusion, encode,
                    load_predictions, per_class_f1, seed_predictions)

overlap_rows = []
for task in TASKS:
    df = load_predictions(task)
    labels = LABELS[task]
    k = len(labels)
    y = encode(df["true"].values, task)
    pred = {m: encode(df[m].values, task) for m in MODELS}

    # Per-class F1 (%); fine-tuned models are averaged over the five seeds.
    f1 = {}
    for m in MODELS:
        if m in FINE_TUNED:
            runs = [per_class_f1(y, encode(seed_predictions(task, m, s, df), task), k) for s in SEEDS]
            f1[MODELS[m]] = np.mean(runs, axis=0) * 100
        else:
            f1[MODELS[m]] = per_class_f1(y, pred[m], k) * 100
    table = pd.DataFrame(f1, index=labels).T
    table["macro-F1"] = table.mean(axis=1)
    table.round(1).to_csv(RESULTS / f"per_class_f1_{task}.csv")

    # Most frequent mistakes: average count per model within each family.
    llm_counts, ft_counts = np.zeros((k, k)), np.zeros((k, k))
    for m in MODELS:
        counts = confusion(y, pred[m], k)[:, :k].astype(float)
        if m in LLMS:
            llm_counts += counts / len(LLMS)
        else:
            ft_counts += counts / len(FINE_TUNED)
    mistakes = pd.DataFrame(
        [(labels[i], labels[j], llm_counts[i, j], ft_counts[i, j])
         for i in range(k) for j in range(k) if i != j],
        columns=["true label", "model said", "LLMs (avg per model)", "fine-tuned (avg per model)"])
    mistakes["total"] = mistakes.iloc[:, 2] + mistakes.iloc[:, 3]
    mistakes = mistakes.sort_values("total", ascending=False).drop(columns="total").head(10).round(1)
    mistakes.to_csv(RESULTS / f"top_confusions_{task}.csv", index=False)

    # Error overlap between the models.
    correct = np.vstack([pred[m] == y for m in MODELS])
    llm_rows = [i for i, m in enumerate(MODELS) if m in LLMS]
    ft_rows = [i for i, m in enumerate(MODELS) if m in FINE_TUNED]
    n_right = correct.sum(axis=0)
    llm_wrong = ~correct[llm_rows].any(axis=0)
    ft_wrong = ~correct[ft_rows].any(axis=0)
    n = len(y)
    overlap_rows += [
        {"task": task, "measure": "all 7 models wrong", "texts": int((n_right == 0).sum()), "share_pct": round(100 * (n_right == 0).mean(), 1)},
        {"task": task, "measure": "all 7 models right", "texts": int((n_right == len(MODELS)).sum()), "share_pct": round(100 * (n_right == len(MODELS)).mean(), 1)},
        {"task": task, "measure": "all LLMs wrong, at least one fine-tuned right", "texts": int((llm_wrong & ~ft_wrong).sum()), "share_pct": round(100 * (llm_wrong & ~ft_wrong).mean(), 1)},
        {"task": task, "measure": "all fine-tuned wrong, at least one LLM right", "texts": int((ft_wrong & ~llm_wrong).sum()), "share_pct": round(100 * (ft_wrong & ~llm_wrong).mean(), 1)},
        {"task": task, "measure": "at least one model right", "texts": int((n_right > 0).sum()), "share_pct": round(100 * (n_right > 0).mean(), 1)},
    ]
    wrong = df[n_right == 0][["text", "true"] + list(MODELS)]
    wrong.to_csv(RESULTS / f"all_models_wrong_{task}.csv", index=False, encoding="utf-8-sig")

    print(f"\n{task}: per-class F1 (%)")
    print(table.round(1).to_string())
    print(mistakes.to_string(index=False))

overlap = pd.DataFrame(overlap_rows)
overlap.to_csv(RESULTS / "error_overlap.csv", index=False)
print("\n", overlap.to_string(index=False))
