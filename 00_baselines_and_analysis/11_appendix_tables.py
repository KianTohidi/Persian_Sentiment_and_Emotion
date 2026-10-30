"""Export the appendix tables (pairwise tests, seed robustness, label changes) to results/appendix_tables.xlsx."""
import pandas as pd

from common import LABEL_DIR, RESULTS, TASKS, read_csv


def format_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


pairs = pd.read_csv(RESULTS / "pairwise_tests.csv")
pairs["p (unadjusted)"] = pairs["p_raw"].map(format_p)
pairs["p (Holm)"] = pairs["p_holm"].map(format_p)
pairs["diff_pp"] = pairs["diff_pp"].round(2)
pairs = pairs.rename(columns={
    "A": "Model A", "B": "Model B", "A_only_right": "Only A correct (texts)", "B_only_right": "Only B correct (texts)",
    "diff_pp": "Accuracy A minus B (pp)", "real_difference": "Significant after Holm"})
pair_columns = ["Model A", "Model B", "Only A correct (texts)", "Only B correct (texts)",
                "Accuracy A minus B (pp)", "p (unadjusted)", "p (Holm)", "Significant after Holm"]
robustness = pd.read_csv(RESULTS / "seed_robustness.csv")

with pd.ExcelWriter(RESULTS / "appendix_tables.xlsx") as writer:
    for task in TASKS:
        pairs[pairs["task"] == task][pair_columns].to_excel(writer, sheet_name=f"C_pairwise_{task}", index=False)
        robustness[robustness["task"] == task].drop(columns="task").to_excel(writer, sheet_name=f"D_seeds_{task}", index=False)
        final = read_csv(LABEL_DIR / f"final_consensus_{task}.csv")
        wrong = final[final["group"] == "all_models_wrong"]
        table = pd.crosstab(wrong["gt"], wrong["consensus"])
        table.index.name = "dataset label (rows) / consensus label (columns)"
        table.to_excel(writer, sheet_name=f"E_labels_{task}")
print("saved", RESULTS / "appendix_tables.xlsx")
