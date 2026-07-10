# Step 1: Upload your file
print("Please upload dataset file: dataset4.csv")
print("\n")

from google.colab import files
uploaded = files.upload()

# Step 2: Import libraries
import pandas as pd

# Step 3: Load the dataset
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name, encoding='utf-8-sig')

# Step 4: Inspect structure
df.head()

# Step 5: Identify columns
text_col = df.columns[0]
emotion_cols_raw = df.columns[1:].tolist()  # Anger, Fear, Happiness, Hatred, Sadness, Wonder

# Step 6: Rename emotion columns to lowercase (professional naming)
LABEL_MAP = {col: col.strip().lower() for col in emotion_cols_raw}
df = df.rename(columns=LABEL_MAP)
emotion_cols = list(LABEL_MAP.values())  # anger, fear, happiness, hatred, sadness, wonder

# Step 7: Threshold for binary presence (>=3 out of 5 votes = emotion present)
THRESHOLD = 3

# Step 8: Count how many emotions meet/exceed the threshold, per row
emotion_count = (df[emotion_cols] >= THRESHOLD).sum(axis=1)

# Step 9: Keep only rows with EXACTLY ONE dominant emotion (single-label cases)
# This excludes rows with zero emotions >= threshold (no clear emotion)
# and rows with multiple emotions >= threshold (ambiguous/multi-label cases),
# to match the single-label annotation scheme used in datasets 1, 2, 3, and 5.
mask = emotion_count == 1
eligible_df = df[mask].reset_index(drop=True)

print(f"Total rows in dataset: {len(df)}")
print(f"Rows with exactly one emotion >= {THRESHOLD} (single dominant emotion): {len(eligible_df)}")
print(f"Rows excluded (0 emotions >= {THRESHOLD}): {(emotion_count == 0).sum()}")
print(f"Rows excluded (2+ emotions >= {THRESHOLD}, ambiguous): {(emotion_count > 1).sum()}")

# Step 10: Random seed for reproducibility
RANDOM_SEED = 42

# Step 11: Sample 150 random rows from eligible rows only
sample_df = eligible_df.sample(n=150, random_state=RANDOM_SEED).reset_index(drop=True)

# Step 12: Derive the single dominant emotion label for each sampled row
def get_dominant_emotion(row):
    for col in emotion_cols:
        if row[col] >= THRESHOLD:
            return col
    return None  # should not occur, given the filtering above

sample_df['emotion'] = sample_df.apply(get_dominant_emotion, axis=1)

# Step 13: Ground truth file — text + single emotion label column
ground_truth_df = sample_df[[text_col, 'emotion']].copy()
ground_truth_df = ground_truth_df.rename(columns={'emotion': 'ground_truth'})

# Step 14: Define accepted annotation labels (sorted alphabetically)
ACCEPTED_LABELS = sorted(emotion_cols)

# Step 15: Annotator files — text only, empty single-label annotation column added
annotator1_df = sample_df[[text_col]].copy()
annotator1_df['annotator1'] = ''

annotator2_df = sample_df[[text_col]].copy()
annotator2_df['annotator2'] = ''

# Step 16: Save all three files
ground_truth_df.to_csv('d4_ground_truth.csv', index=False, encoding='utf-8-sig')
annotator1_df.to_csv('d4_annotator1.csv', index=False, encoding='utf-8-sig')
annotator2_df.to_csv('d4_annotator2.csv', index=False, encoding='utf-8-sig')

# Step 17: Download
files.download('d4_ground_truth.csv')
files.download('d4_annotator1.csv')
files.download('d4_annotator2.csv')

# Step 18: Instructions for annotators
print("Files generated successfully.")
print(f"Accepted annotation labels: {ACCEPTED_LABELS}")
print("Please instruct annotators to fill in the empty column using EXACTLY one of these values")
print(f"(lowercase, no extra spaces) for each of the 150 rows: {', '.join(ACCEPTED_LABELS)}.")