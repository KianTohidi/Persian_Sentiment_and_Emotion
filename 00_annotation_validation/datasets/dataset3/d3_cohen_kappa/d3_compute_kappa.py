# Step 1: Instructions before upload
print("=" * 60)
print("READY TO CALCULATE INTER-ANNOTATOR AGREEMENT")
print("=" * 60)
print("\nPlease upload exactly 3 files, named precisely as follows:")
print("  1. d3_ground_truth.csv   -> original file, unmodified")
print("  2. d3_annotator1.csv     -> fully filled in by Annotator 1")
print("  3. d3_annotator2.csv     -> fully filled in by Annotator 2")
print("\nTIP: You can select all 3 files at once in the upload window —")
print("hold Ctrl (Windows/Linux) or Cmd (Mac) while clicking each file,")
print("then click Open to upload them together.")
print("\nBefore uploading, please make sure:")
print("  - All 150 rows in BOTH annotator files have been labeled")
print("    (no empty cells left in the 'annotator1' / 'annotator2' column).")
print("  - Each label is EXACTLY one of: anger, fear, happiness, hate, other, sadness, surprise")
print("    (lowercase, no extra spaces, no typos, no other categories).")
print("  - No rows were added, deleted, reordered, or edited in the text column.")
print("  - File names were not changed after download.")
print("\nIf any of the above isn't true, please fix it before uploading —")
print("the script will stop and show an error if something doesn't match.")
print("=" * 60)

from google.colab import files
uploaded = files.upload()

# Step 2: Import libraries
import pandas as pd
import re
from sklearn.metrics import cohen_kappa_score

# Step 3: Match uploaded files to expected names, even if Colab
# auto-renamed them (e.g. "d3_annotator1 (1).csv") due to a naming clash
expected_files = ['d3_ground_truth.csv', 'd3_annotator1.csv', 'd3_annotator2.csv']
uploaded_names = list(uploaded.keys())

def find_uploaded_match(expected_name, uploaded_names):
    base = expected_name.replace('.csv', '')
    # Matches "d3_annotator1.csv", "d3_annotator1 (1).csv", "d3_annotator1(2).csv", etc.
    pattern = re.compile(rf"^{re.escape(base)}(\s*\(\d+\))?\.csv$")
    matches = [name for name in uploaded_names if pattern.match(name)]
    if len(matches) > 1:
        raise ValueError(f"Multiple files matched '{expected_name}': {matches}. "
                          f"Please re-upload with only one copy of each file.")
    return matches[0] if matches else None

file_map = {}
missing = []
for expected in expected_files:
    match = find_uploaded_match(expected, uploaded_names)
    if match is None:
        missing.append(expected)
    else:
        file_map[expected] = match

if missing:
    raise ValueError(f"Missing expected file(s): {missing}. "
                      f"Please upload exactly: {expected_files}. "
                      f"Files received: {uploaded_names}")

# Step 4: Load the files (using matched names, in case of "(1)" suffixes)
gt_df = pd.read_csv(file_map['d3_ground_truth.csv'], encoding='utf-8-sig')
a1_df = pd.read_csv(file_map['d3_annotator1.csv'], encoding='utf-8-sig')
a2_df = pd.read_csv(file_map['d3_annotator2.csv'], encoding='utf-8-sig')

# Step 5: Sanity check — same number of rows
if not (len(gt_df) == len(a1_df) == len(a2_df) == 150):
    raise ValueError(f"Row count mismatch: ground_truth={len(gt_df)}, "
                      f"annotator1={len(a1_df)}, annotator2={len(a2_df)}. "
                      f"All files must have 150 rows.")

# Step 6: Sanity check — same texts in same order across all files
text_col_gt = gt_df.columns[0]
text_col_a1 = a1_df.columns[0]
text_col_a2 = a2_df.columns[0]

if not (gt_df[text_col_gt].equals(a1_df[text_col_a1]) and
        gt_df[text_col_gt].equals(a2_df[text_col_a2])):
    raise ValueError("Text mismatch detected: rows are not aligned across the three files. "
                      "Make sure no rows were added, removed, or reordered.")

# Step 7: Validate annotation labels (including missing/empty check)
ACCEPTED_LABELS = ['anger', 'fear', 'happiness', 'hate', 'other', 'sadness', 'surprise']

def validate_labels(df, label_col, name):
    # Check for empty/missing cells first
    missing_mask = df[label_col].isna() | (df[label_col].astype(str).str.strip() == '')
    if missing_mask.any():
        missing_rows = df.loc[missing_mask, [df.columns[0]]]
        print(f"\n⚠️ Empty label(s) found in {name}:")
        print(missing_rows)
        raise ValueError(f"{name} has {missing_mask.sum()} unfilled row(s). "
                          f"All 150 rows must be labeled before uploading.")

    # Check for invalid label values
    cleaned = df[label_col].astype(str).str.strip().str.lower()
    invalid_mask = ~cleaned.isin(ACCEPTED_LABELS)
    if invalid_mask.any():
        bad_rows = df.loc[invalid_mask, [df.columns[0], label_col]]
        print(f"\n⚠️ Invalid label(s) found in {name}:")
        print(bad_rows)
        raise ValueError(f"{name} contains {invalid_mask.sum()} invalid label(s). "
                          f"Accepted values: {ACCEPTED_LABELS}")
    return cleaned

a1_labels = validate_labels(a1_df, 'annotator1', 'annotator1 file')
a2_labels = validate_labels(a2_df, 'annotator2', 'annotator2 file')
gt_labels = validate_labels(gt_df, 'ground_truth', 'ground truth file')

# Step 8: Compute Cohen's kappa
kappa_a1_a2 = cohen_kappa_score(a1_labels, a2_labels)
kappa_a1_gt = cohen_kappa_score(a1_labels, gt_labels)
kappa_a2_gt = cohen_kappa_score(a2_labels, gt_labels)

# Step 9: Percent agreement (simple IAA, complements kappa)
agreement_a1_a2 = (a1_labels == a2_labels).mean() * 100
agreement_a1_gt = (a1_labels == gt_labels).mean() * 100
agreement_a2_gt = (a2_labels == gt_labels).mean() * 100

# Step 10: Report results
print("=" * 50)
print("Inter-Annotator Agreement Results")
print("=" * 50)
print(f"Annotator1 vs Annotator2:")
print(f"  Cohen's kappa: {kappa_a1_a2:.3f}")
print(f"  Percent agreement: {agreement_a1_a2:.1f}%\n")

print(f"Annotator1 vs Ground Truth:")
print(f"  Cohen's kappa: {kappa_a1_gt:.3f}")
print(f"  Percent agreement: {agreement_a1_gt:.1f}%\n")

print(f"Annotator2 vs Ground Truth:")
print(f"  Cohen's kappa: {kappa_a2_gt:.3f}")
print(f"  Percent agreement: {agreement_a2_gt:.1f}%")
print("=" * 50)

# Step 11: Combine everything into one summary CSV for reference
summary_df = gt_df[[text_col_gt]].copy()
summary_df['ground_truth'] = gt_labels
summary_df['annotator1'] = a1_labels
summary_df['annotator2'] = a2_labels
summary_df.to_csv('d3_combined_annotations.csv', index=False, encoding='utf-8-sig')
files.download('d3_combined_annotations.csv')