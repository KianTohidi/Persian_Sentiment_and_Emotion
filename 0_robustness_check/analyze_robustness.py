"""
analyze_robustness.py

Batch(20) vs single-instance(1) robustness check analysis.
Run once per CSV (one CSV per model x task, 8 total: 4 models x
{Sentiment, Emotion}). Each CSV's first four columns must be, in order:
text, ground_truth_label, batch20_label, batch1_label.

Computes accuracy and macro F1 for both conditions against ground
truth, Cohen's kappa between the two conditions, McNemar's exact test
on paired correctness, and reports the disagreement count and per-
condition classification reports.

Usage (Google Colab):
    Run the cell, then use the upload dialog to select one CSV.
    Repeat once per CSV (8 times total) to reproduce every row of
    robustness_check_summary.csv / Tables 4.3a-b in the paper.
"""

!pip install -q scikit-learn pandas statsmodels

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar
from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]

df = pd.read_csv(filename)
text_col, gt_col, pred1_col, pred2_col = df.columns[:4]

y_true = df[gt_col].astype(str)
y_pred1 = df[pred1_col].astype(str)
y_pred2 = df[pred2_col].astype(str)

acc1 = accuracy_score(y_true, y_pred1)
acc2 = accuracy_score(y_true, y_pred2)
f1_1 = f1_score(y_true, y_pred1, average='macro', zero_division=0)
f1_2 = f1_score(y_true, y_pred2, average='macro', zero_division=0)
kappa = cohen_kappa_score(y_pred1, y_pred2)

print(f"Accuracy ({pred1_col} vs {gt_col}): {acc1:.4f}")
print(f"Accuracy ({pred2_col} vs {gt_col}): {acc2:.4f}")
print(f"Macro F1 ({pred1_col} vs {gt_col}): {f1_1:.4f}")
print(f"Macro F1 ({pred2_col} vs {gt_col}): {f1_2:.4f}")
print(f"Cohen's Kappa ({pred1_col} vs {pred2_col}): {kappa:.4f}")

correct1 = (y_pred1 == y_true)
correct2 = (y_pred2 == y_true)

both_correct = ((correct1) & (correct2)).sum()
only1_correct = ((correct1) & (~correct2)).sum()
only2_correct = ((~correct1) & (correct2)).sum()
both_wrong = ((~correct1) & (~correct2)).sum()

table = [[both_correct, only1_correct], [only2_correct, both_wrong]]
result = mcnemar(table, exact=True)
print(f"\nMcNemar's test statistic: {result.statistic:.4f}")
print(f"McNemar's test p-value: {result.pvalue:.4f}")

disagreements = (y_pred1 != y_pred2).sum()
print(f"\nDisagreement count: {disagreements} out of {len(df)} texts differ between {pred1_col} and {pred2_col}")

print(f"\nClassification report for {pred1_col} vs {gt_col}:")
print(classification_report(y_true, y_pred1, zero_division=0))

print(f"Classification report for {pred2_col} vs {gt_col}:")
print(classification_report(y_true, y_pred2, zero_division=0))