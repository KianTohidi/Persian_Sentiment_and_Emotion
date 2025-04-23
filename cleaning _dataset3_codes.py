# Data Cleaning and Processing Pipeline
import pandas as pd
from google.colab import files


# Step 1: Upload and Load Datasets
print("Please upload both dataset files (dataset3_part1.csv and dataset3_part2.csv):")
uploaded = files.upload()  # Upload both files


# Load both datasets using UTF-8-SIG encoding
df1 = pd.read_csv("dataset3_part1.csv", header=None, encoding='utf-8-sig')
df2 = pd.read_csv("dataset3_part2.csv", header=None, encoding='utf-8-sig')


# Step 2: Initial Dataset Analysis
print("\n===== Initial Dataset Analysis =====")
print(f"Rows in part 1: {len(df1)}")
print(f"Rows in part 2: {len(df2)}")
print(f"Total rows before merging: {len(df1) + len(df2)}")





# Step 3: Merge Datasets
merged_df = pd.concat([df1, df2], ignore_index=True)
print(f"\nTotal rows after merge: {len(merged_df)}")




# Step 4: Add Column Names
merged_df.columns = ['text', 'emotion']
print("\n===== Dataset Preview =====")
merged_df.head()


# Step 5: Clean the Dataset
print("\n===== Cleaning Process =====")


# Remove empty or missing text entries
rows_before = len(merged_df)
merged_df = merged_df.dropna(subset=['text'])
merged_df = merged_df[merged_df['text'].astype(str).str.strip() != '']
rows_after = len(merged_df)
print(f"✅ Removed empty text entries: {rows_before - rows_after}")



# Remove duplicate text entries
rows_before = len(merged_df)
merged_df = merged_df.drop_duplicates(subset='text')
rows_after = len(merged_df)
print(f"✅ Removed duplicate texts: {rows_before - rows_after}")




# Remove short texts (less than 10 characters)
rows_before = len(merged_df)
merged_df = merged_df[merged_df['text'].astype(str).str.len() >= 10]
rows_after = len(merged_df)
print(f"✅ Removed short texts: {rows_before - rows_after}")




# Remove long texts (more than 100 words)
rows_before = len(merged_df)
merged_df = merged_df[merged_df['text'].astype(str).apply(lambda x: len(x.split())) <= 100]
rows_after = len(merged_df)
print(f"✅ Removed overly long texts (>100 words): {rows_before - rows_after}")



# Remove 'other' emotion labels
rows_before = len(merged_df)
merged_df = merged_df[merged_df['emotion'].astype(str).str.strip().str.lower() != 'other']
rows_after = len(merged_df)
print(f"✅ Removed 'other' emotion labels: {rows_before - rows_after}")




# Checking if “OTHER” is removed

merged_df.head()


# Step 6: Standardize Emotion Labels
# Ensure all labels are lowercase and stripped first
merged_df['emotion'] = merged_df['emotion'].astype(str).str.strip().str.lower()


# Checking lowercase

merged_df.head()


# Define mapping from verb-like to noun labels
emotion_mapping = {
    'sad': 'sadness',
    'hate': 'hate',
    'fear': 'fear',
    'angry': 'anger',
    'surprise': 'surprise',
    'happy': 'happiness'
}


# Apply mapping
merged_df['emotion'] = merged_df['emotion'].replace(emotion_mapping)


# Check unique emotion labels after standardization
print("\n===== Quality Control =====")
print("Unique emotion labels after standardization:")
print(merged_df['emotion'].unique())



['sadness' 'happiness' 'surprise' 'fear' 'hate' 'anger']


# visual checking

merged_df.head()


# Step 7: Final Dataset Information
print("\n===== Final Dataset Information =====")
print(f"Final dataset size: {len(merged_df)}")
print("\nFinal dataset preview:")
merged_df.head()




# Step 8: Save the Cleaned Dataset
cleaned_file = "dataset3_cleaned.csv"
merged_df.to_csv(cleaned_file, index=False, encoding="utf-8-sig")
print(f"\nCleaned dataset saved as '{cleaned_file}'")
files.download(cleaned_file)
print("Download initiated.")




