"""Analyse the consensus labels of the label check.

For texts on which the annotators agreed, the agreed label is kept; for the others, the consensus label
from label_check/consensus_sheet.xlsx is used. The consensus labels are compared with the dataset labels
(Wilson 95% CIs, Fisher's exact test) and with the predictions of the seven models.
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from common import LABEL_DIR, LABELS, MODELS, TASKS, load_predictions, read_csv, wilson

summary = []
for task in TASKS:
    allowed = set(LABELS[task] + ["other_unclear"])
    d = read_csv(LABEL_DIR / f"item_level_results_{task}.csv")
    d = d.merge(read_csv(LABEL_DIR / f"{task}_key.csv")[["item_id", "test_row"]], on="item_id")
    d["test_row"] = d["test_row"].astype(int)

    sheet = pd.read_excel(LABEL_DIR / "consensus_sheet.xlsx", sheet_name=task, dtype=str, keep_default_na=False)
    sheet = sheet[["item_id", "consensus_label"]].rename(columns={"consensus_label": "discussed_label"})
    d = d.merge(sheet, on="item_id", how="left")
    d["discussed"] = d["a1"] != d["a2"]
    d["consensus"] = np.where(d["discussed"], d["discussed_label"].fillna("").str.strip().str.lower(), d["a1"])
    missing = d["discussed"] & ~d["consensus"].isin(allowed)
    if missing.any():
        print(f"warning: {int(missing.sum())} discussed texts without a valid consensus label were skipped")
        d = d[~missing].copy()

    predictions = load_predictions(task)
    for m in MODELS:
        d[m] = predictions[m].values[d["test_row"].values]
    wrong = d[d["group"] == "all_models_wrong"]
    control = d[d["group"] == "all_models_right"]
    discussed = d[d["discussed"]]

    print(f"\n===== {task}: {len(wrong)} texts wrong for all models + {len(control)} control texts =====")
    print(f"discussed: {len(discussed)} | consensus equals annotator 1: {int((discussed['consensus'] == discussed['a1']).sum())}"
          f" | annotator 2: {int((discussed['consensus'] == discussed['a2']).sum())}"
          f" | other: {int(((discussed['consensus'] != discussed['a1']) & (discussed['consensus'] != discussed['a2'])).sum())}")

    matches = {}
    for name, part in [("all texts", d), ("all_models_wrong", wrong), ("control", control)]:
        k, n = int((part["consensus"] == part["gt"]).sum()), len(part)
        low, high = wilson(k, n)
        matches[name] = (k, n)
        print(f"consensus = dataset label, {name}: {k}/{n} = {100 * k / n:.1f}% (95% CI {100 * low:.1f}-{100 * high:.1f})")
        summary.append({"task": task, "measure": f"consensus equals dataset label ({name})", "n": n,
                        "matches": k, "pct": round(100 * k / n, 1), "ci_low": round(100 * low, 1),
                        "ci_high": round(100 * high, 1)})
    (kw, nw), (kc, nc) = matches["all_models_wrong"], matches["control"]
    p_value = fisher_exact([[kw, nw - kw], [kc, nc - kc]])[1]
    print(f"Fisher exact test, wrong group vs control: p = {p_value:.3g}")
    summary.append({"task": task, "measure": "Fisher exact test (wrong vs control)", "p_value": float(f"{p_value:.3g}")})

    print("\nTexts wrong for all models, by dataset label:")
    by_label = pd.DataFrame({
        "texts": wrong.groupby("gt").size(),
        "consensus = dataset label": wrong[wrong["consensus"] == wrong["gt"]].groupby("gt").size(),
    }).fillna(0).astype(int)
    by_label["% confirmed"] = (100 * by_label["consensus = dataset label"] / by_label["texts"]).round(1)
    print(by_label.to_string())
    print("\nDataset label (rows) vs consensus label (columns):")
    print(pd.crosstab(wrong["gt"], wrong["consensus"]).to_string())

    changed = wrong[wrong["consensus"] != wrong["gt"]]
    if len(changed):
        n_models = np.column_stack([(changed[m] == changed["consensus"]).values for m in MODELS]).sum(axis=1)
        share = {MODELS[m]: round(100 * float((wrong[m] == wrong["consensus"]).mean()), 1) for m in MODELS}
        print(f"\nConsensus differs from the dataset label in {len(changed)} of {len(wrong)} texts.")
        print(f"At least one model predicted the consensus label: {int((n_models >= 1).sum())} | "
              f"majority (4+ of 7): {int((n_models >= 4).sum())} | mean models matching: {n_models.mean():.2f}")
        print("Share of each model's predictions equal to the consensus label (%):", share)
        summary.append({"task": task, "measure": "texts with consensus different from dataset label", "n": len(changed),
                        "at_least_1_model": int((n_models >= 1).sum()), "majority_models": int((n_models >= 4).sum()),
                        "mean_models_matching": round(float(n_models.mean()), 2)})

    columns = ["item_id", "text", "group", "gt", "a1", "a2", "discussed", "consensus"] + list(MODELS)
    d[columns].to_csv(LABEL_DIR / f"final_consensus_{task}.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(summary).to_csv(LABEL_DIR / "consensus_summary.csv", index=False)
