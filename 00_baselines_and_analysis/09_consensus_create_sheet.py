"""Create the consensus sheet: only the texts on which the two annotators disagreed, without dataset labels."""
import re

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from common import LABEL_DIR, LABELS, TASKS, read_csv


def excel_safe(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text))


if (LABEL_DIR / "consensus_sheet.xlsx").exists():
    raise SystemExit("consensus_sheet.xlsx already exists; delete it first to regenerate a blank sheet.")

workbook = Workbook()
workbook.remove(workbook.active)
key_rows = []

for task in TASKS:
    d = read_csv(LABEL_DIR / f"item_level_results_{task}.csv")
    key = read_csv(LABEL_DIR / f"{task}_key.csv")[["item_id", "test_row"]]
    disagreements = d[d["a1"] != d["a2"]].merge(key, on="item_id").sort_values("item_id").reset_index(drop=True)

    ws = workbook.create_sheet(task)
    ws.sheet_view.rightToLeft = True
    ws.append(["text", "annotator_1_label", "annotator_2_label", "consensus_label", "notes (optional)", "item_id"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
    for row, r in enumerate(disagreements.itertuples(index=False), start=2):
        cell = ws.cell(row=row, column=1, value=excel_safe(r.text))
        cell.data_type = "s"
        cell.alignment = Alignment(horizontal="right", vertical="top", wrap_text=True)
        ws.cell(row=row, column=2, value=r.a1)
        ws.cell(row=row, column=3, value=r.a2)
        ws.cell(row=row, column=6, value=r.item_id)
        key_rows.append({"task": task, "item_id": r.item_id, "text": r.text, "ground_truth": r.gt,
                         "group": r.group, "test_row": r.test_row,
                         "annotator_1_label": r.a1, "annotator_2_label": r.a2})
    if len(disagreements):
        validation = DataValidation(type="list", formula1='"' + ",".join(LABELS[task] + ["other_unclear"]) + '"',
                                    allow_blank=True)
        ws.add_data_validation(validation)
        validation.add(f"D2:D{len(disagreements) + 1}")
    ws.column_dimensions["A"].width = 90
    for column in "BCD":
        ws.column_dimensions[column].width = 22
    ws.column_dimensions["E"].width = 30
    ws.column_dimensions["F"].width = 10
    ws.freeze_panes = "A2"
    print(f"{task}: {len(disagreements)} texts to discuss")

workbook.save(LABEL_DIR / "consensus_sheet.xlsx")
pd.DataFrame(key_rows).to_csv(LABEL_DIR / "consensus_key.csv", index=False, encoding="utf-8-sig")
