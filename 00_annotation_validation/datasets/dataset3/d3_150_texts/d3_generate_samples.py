# Step 1: Upload your file
print("Please upload dataset file: dataset3.csv")
print("\n")

from google.colab import files
uploaded = files.upload()

# Step 2: Import libraries
import pandas as pd

# Step 3: Load the dataset
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name, encoding='utf-8-sig')

# Step 3b: Fix column titles (lowercase instead of Title case)
df.columns = ['text', 'emotion']

# Step 4: Inspect structure
df.head()

# Step 5: Random seed for reproducibility
RANDOM_SEED = 42

# Step 6: Sample 150 random rows (same rows every run)
sample_df = df.sample(n=150, random_state=RANDOM_SEED).reset_index(drop=True)
text_col = sample_df.columns[0]
label_col = sample_df.columns[1]

# Step 6b: Normalize original labels (all-caps, some renamed) to accepted labels
LABEL_MAP = {
    'SAD': 'sadness',
    'HAPPY': 'happiness',
    'FEAR': 'fear',
    'SURPRISE': 'surprise',
    'HATE': 'hate',
    'ANGRY': 'anger',
    'OTHER': 'other'
}
sample_df[label_col] = sample_df[label_col].map(LABEL_MAP)

# Step 7: Define accepted annotation labels (sorted alphabetically)
ACCEPTED_LABELS = ['anger', 'fear', 'happiness', 'hate', 'other', 'sadness', 'surprise']

# Step 8: Ground truth file — keeps text + original label, no column removal
ground_truth_df = sample_df[[text_col, label_col]].copy()
ground_truth_df = ground_truth_df.rename(columns={label_col: 'ground_truth'})

# Step 9: Annotator files — text only, second column removed, empty annotation column added
annotator1_df = sample_df[[text_col]].copy()
annotator1_df['annotator1'] = ''

annotator2_df = sample_df[[text_col]].copy()
annotator2_df['annotator2'] = ''

# Step 10: Save all three files
ground_truth_df.to_csv('d3_ground_truth.csv', index=False, encoding='utf-8-sig')
annotator1_df.to_csv('d3_annotator1.csv', index=False, encoding='utf-8-sig')
annotator2_df.to_csv('d3_annotator2.csv', index=False, encoding='utf-8-sig')

# Step 11: Download
files.download('d3_ground_truth.csv')
files.download('d3_annotator1.csv')
files.download('d3_annotator2.csv')

# Step 12: Instructions for annotators
print("Files generated successfully.")
print(f"Accepted annotation labels: {ACCEPTED_LABELS}")
print("Please instruct annotators to fill in the empty column using EXACTLY one of these values")
print("(lowercase, no extra spaces) for each of the 150 rows: anger, fear, happiness, hate, other, sadness, surprise.")