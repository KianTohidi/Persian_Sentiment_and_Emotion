"""Create the blind annotation sheets for the label check.

Selects the texts that all seven models classified wrongly and an equal number of control texts that
all models classified correctly (same dataset-label distribution, seed 42), shuffles them, and writes
one sheet per annotator without dataset labels plus a key file.
"""
import re

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from common import LABEL_DIR, LABELS, MODELS, TASKS, load_predictions

SEED = 42
PREFIX = {"sentiment": "S", "emotion": "E"}
LABEL_DIR.mkdir(parents=True, exist_ok=True)
if any(LABEL_DIR.glob("*_annotator*.xlsx")):
    raise SystemExit("Annotation sheets already exist in label_check/; delete them first to regenerate blank sheets.")


def excel_safe(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))


def write_sheet(path, items, header, allowed):
    wb = Workbook()
    ws = wb.active
    ws.title = "annotation"
    ws.sheet_view.rightToLeft = True
    ws.append(["text", header, "item_id"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
    for row, (text, item_id) in enumerate(zip(items["text"], items["item_id"]), start=2):
        cell = ws.cell(row=row, column=1, value=excel_safe(text))
        cell.data_type = "s"  # store as plain text, never as a formula
        cell.alignment = Alignment(horizontal="right", vertical="top", wrap_text=True)
        ws.cell(row=row, column=3, value=item_id)
    validation = DataValidation(type="list", formula1='"' + ",".join(allowed) + '"', allow_blank=True)
    ws.add_data_validation(validation)
    validation.add(f"B2:B{len(items) + 1}")
    ws.column_dimensions["A"].width = 90
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 10
    ws.freeze_panes = "A2"
    wb.save(path)


for task in TASKS:
    df = load_predictions(task)
    n_right = np.column_stack([(df[m] == df["true"]).values for m in MODELS]).sum(axis=1)
    rng = np.random.default_rng(SEED)

    wrong_idx = np.where(n_right == 0)[0]
    control_idx = []
    for label in LABELS[task]:
        needed = int((df["true"].values[wrong_idx] == label).sum())
        pool = np.where((n_right == len(MODELS)) & (df["true"].values == label))[0]
        control_idx += list(rng.choice(pool, min(needed, len(pool)), replace=False))

    key = pd.concat([
        pd.DataFrame({"text": df["text"].values[wrong_idx], "ground_truth": df["true"].values[wrong_idx],
                      "group": "all_models_wrong", "test_row": wrong_idx}),
        pd.DataFrame({"text": df["text"].values[control_idx], "ground_truth": df["true"].values[control_idx],
                      "group": "all_models_right", "test_row": control_idx}),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    key["item_id"] = [f"{PREFIX[task]}{i + 1:03d}" for i in range(len(key))]
    key = key[["text", "ground_truth", "item_id", "group", "test_row"]]

    allowed = LABELS[task] + ["other_unclear"]
    for annotator in (1, 2):
        write_sheet(LABEL_DIR / f"{task}_annotator{annotator}.xlsx", key, f"annotator_{annotator}_label", allowed)
    key.to_csv(LABEL_DIR / f"{task}_key.csv", index=False, encoding="utf-8-sig")
    print(f"{task}: {len(wrong_idx)} texts wrong for all models + {len(control_idx)} control texts")
