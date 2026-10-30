"""Collect the two independent annotations and compute inter-annotator agreement (before discussion)."""
import pandas as pd

from common import LABEL_DIR, LABELS, TASKS, kappa, kappa_ci, read_csv


def read_annotator(task, annotator):
    sheet = pd.read_excel(LABEL_DIR / f"{task}_annotator{annotator}.xlsx", dtype=str, keep_default_na=False)
    column = f"annotator_{annotator}_label"
    return sheet[["item_id", column]].rename(columns={column: f"a{annotator}"})


summary = []
for task in TASKS:
    allowed = set(LABELS[task] + ["other_unclear"])
    key = read_csv(LABEL_DIR / f"{task}_key.csv").rename(columns={"ground_truth": "gt"})
    d = key[["item_id", "text", "gt", "group"]]
    for annotator in (1, 2):
        d = d.merge(read_annotator(task, annotator), on="item_id", how="left")
        d[f"a{annotator}"] = d[f"a{annotator}"].fillna("").str.strip().str.lower()
        invalid = ~d[f"a{annotator}"].isin(allowed)
        print(f"{task}: annotator {annotator} labelled {int((~invalid).sum())}/{len(d)} texts")
    d = d[d["a1"].isin(allowed) & d["a2"].isin(allowed)]
    d.to_csv(LABEL_DIR / f"item_level_results_{task}.csv", index=False, encoding="utf-8-sig")

    for name, part in [("all texts", d), ("all_models_wrong", d[d["group"] == "all_models_wrong"]),
                       ("control", d[d["group"] == "all_models_right"])]:
        agreement = (part["a1"] == part["a2"]).mean() * 100
        low, high = kappa_ci(part["a1"], part["a2"])
        value = kappa(part["a1"], part["a2"])
        summary.append({"task": task, "group": name, "n": len(part), "agreement_pct": round(agreement, 1),
                        "kappa": round(value, 3), "ci_low": round(low, 3), "ci_high": round(high, 3)})

summary = pd.DataFrame(summary)
summary.to_csv(LABEL_DIR / "annotation_summary.csv", index=False)
print(summary.to_string(index=False))
