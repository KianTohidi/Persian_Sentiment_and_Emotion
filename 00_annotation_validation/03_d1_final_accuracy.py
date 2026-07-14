# Step 1: Instructions before upload
print("=" * 60)
print("READY TO CALCULATE FINAL LABELING ACCURACY")
print("=" * 60)
print("\nPlease upload exactly 1 file:")
print("  d1_combined_annotations.csv")
print("\nBefore uploading, please make sure:")
print("  - The 'consensus' column is fully filled in for ALL 150 rows")
print("    (this is the final label the two annotators agreed on after")
print("    discussing every disagreement and re-checking the guidelines).")
print("  - Each consensus label is EXACTLY one of: positive, negative, neutral")
print("    (lowercase, no extra spaces, no typos, no other categories).")
print("  - No rows were added, deleted, reordered, or edited.")
print("\nIf any of the above isn't true, please fix it before uploading —")
print("the script will stop and show an error if something doesn't match.")
print("=" * 60)

from google.colab import files
uploaded = files.upload()

# Step 2: Import libraries
import pandas as pd

# Step 3: Load the uploaded file
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name, encoding='utf-8-sig')

# Step 4: Sanity check — required columns exist
required_cols = ['ground_truth', 'annotator1', 'annotator2', 'consensus']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected column(s): {missing_cols}. "
                      f"Found columns: {list(df.columns)}")

# Step 5: Sanity check — row count
if len(df) != 150:
    raise ValueError(f"Expected 150 rows, found {len(df)}. "
                      f"Make sure no rows were added or removed.")

# Step 6: Validate the consensus column (including missing/empty check)
ACCEPTED_LABELS = ['positive', 'negative', 'neutral']

def validate_labels(df, label_col, name):
    missing_mask = df[label_col].isna() | (df[label_col].astype(str).str.strip() == '')
    if missing_mask.any():
        print(f"\n⚠️ Empty label(s) found in '{name}':")
        print(df.loc[missing_mask, [df.columns[0]]])
        raise ValueError(f"'{name}' has {missing_mask.sum()} unfilled row(s). "
                          f"All 150 consensus labels must be filled before uploading.")

    cleaned = df[label_col].astype(str).str.strip().str.lower()
    invalid_mask = ~cleaned.isin(ACCEPTED_LABELS)
    if invalid_mask.any():
        print(f"\n⚠️ Invalid label(s) found in '{name}':")
        print(df.loc[invalid_mask, [df.columns[0], label_col]])
        raise ValueError(f"'{name}' contains {invalid_mask.sum()} invalid label(s). "
                          f"Accepted values: {ACCEPTED_LABELS}")
    return cleaned

consensus_labels = validate_labels(df, 'consensus', 'consensus column')
ground_truth_labels = df['ground_truth'].astype(str).str.strip().str.lower()

# Step 7: Compute final labeling accuracy — consensus vs. ground truth
matches = (consensus_labels == ground_truth_labels)
accuracy_count = matches.sum()
accuracy_percentage = matches.mean() * 100

# Step 8: how disagreements were resolved
a1 = df['annotator1'].astype(str).str.strip().str.lower()
a2 = df['annotator2'].astype(str).str.strip().str.lower()
disagreement_mask = a1 != a2
n_disagreements = disagreement_mask.sum()

if n_disagreements > 0:
    resolved_to_a1 = (consensus_labels[disagreement_mask] == a1[disagreement_mask]).sum()
    resolved_to_a2 = (consensus_labels[disagreement_mask] == a2[disagreement_mask]).sum()
    resolved_to_neither = n_disagreements - resolved_to_a1 - resolved_to_a2
else:
    resolved_to_a1 = resolved_to_a2 = resolved_to_neither = 0

# Step 9: Report results
print("=" * 50)
print("Final Labeling Accuracy — Consensus vs. Ground Truth")
print("=" * 50)
print(f"Total texts: {len(df)}")
print(f"Matches: {accuracy_count}")
print(f"Accuracy: {accuracy_percentage:.2f}%\n")

print(f"Annotator disagreements (before adjudication): {n_disagreements} "
      f"({n_disagreements / len(df) * 100:.1f}% of texts)")
if n_disagreements > 0:
    print(f"  Resolved to Annotator 1's original label: {resolved_to_a1}")
    print(f"  Resolved to Annotator 2's original label: {resolved_to_a2}")
    print(f"  Resolved to a third, different label:     {resolved_to_neither}")
print("=" * 50)

# Step 10: Save final results file
df['match_ground_truth'] = matches
df.to_csv('d1_final_accuracy_results.csv', index=False, encoding='utf-8-sig')
files.download('d1_final_accuracy_results.csv')