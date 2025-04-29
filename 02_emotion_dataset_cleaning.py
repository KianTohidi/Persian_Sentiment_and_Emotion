"""
Persian Emotion Detection Dataset Cleaning Part 1
-------------------------------------------
This script cleans and preprocesses the Persian social media and e-commerce dataset for emotion detection.
Input: Raw TSV files with social media and e-commerce entries and emotion labels
Output: Cleaned CSV with standardized text and emotion labels
Dependencies: pandas (tested with version 2.2.2)
"""


# Step 1: Import pandas
import pandas as pd


# Step 2: Upload the dataset files
# IMPORTANT: Before running this cell:
# 1. Download the TSV dataset files from the GitHub website (link in the README).
# 2. To do this, on the website link, click the "dataset" folder. Then you will see two TSV files there.
# 3. Click on one of them, and then on the right, you can see the download symbol. Click "Download raw file". Do the same for the other file.
# 4. Ensure both 'train.tsv' and 'test.tsv' are available for upload. During uploading the files, in order to upload both files, press the "Ctrl" button on the keyboard, then select both files to be uploaded.
from google.colab import files
uploaded = files.upload()


# Step 3: Load the TSV files using UTF-8-SIG encoding (important for Persian text)
df1 = pd.read_csv('train.tsv', sep='\t', encoding="utf-8-sig", header=None, names=['text', 'emotion'])
df2 = pd.read_csv('test.tsv', sep='\t', encoding="utf-8-sig", header=None, names=['text', 'emotion'])
print("Train dataset rows:", len(df1))
print("Test dataset rows:", len(df2))
print("\n")

# View first few rows of train dataset
df1.head()


# Step 4: Merge datasets for unified processing
merged_df = pd.concat([df1, df2], ignore_index=True)
print("Total rows after merge:", len(merged_df))

# View first few rows of merged dataset
merged_df.head()


# Step 5: Remove empty or missing text entries
before_empty = len(merged_df)
merged_df = merged_df.dropna(subset=['text'])
merged_df = merged_df[merged_df['text'].astype(str).str.strip() != '']
after_empty = len(merged_df)
print("Removed empty texts:", before_empty - after_empty)
print("Rows after removing empty texts:", len(merged_df))


# Step 6: Remove duplicate text entries to avoid bias
before_duplicates = len(merged_df)
merged_df = merged_df.drop_duplicates(subset='text')
after_duplicates = len(merged_df)
print("Removed duplicate texts:", before_duplicates - after_duplicates)
print("Rows after removing duplicate texts:", len(merged_df))


# Step 7: Apply minimum text length threshold of 10 characters
# Very short texts typically don't contain enough information for emotion detection
before_short = len(merged_df)
merged_df = merged_df[merged_df['text'].astype(str).str.len() >= 10]
after_short = len(merged_df)
print("Removed short texts:", before_short - after_short)
print("Remaining rows after removing short texts:", len(merged_df))


# Step 8: Remove entries with more than 100 words
# Extremely long texts might be outliers or not typical social media content
before_long = len(merged_df)
merged_df = merged_df[merged_df['text'].apply(lambda x: len(str(x).split()) <= 100)]
after_long = len(merged_df)
print("Removed long texts:", before_long - after_long)
print("Remaining rows after removing long texts:", len(merged_df))


# Step 9: Standardize emotion labels for consistency
# Ensure all labels are lowercase and stripped
merged_df['emotion'] = merged_df['emotion'].astype(str).str.strip().str.lower()

# Define mapping from verb-like to noun labels
emotion_mapping = {
    'sad': 'sadness',
    'hate': 'hate',
    'fear': 'fear',
    'angry': 'anger',
    'surprise': 'surprise',
    'happy': 'happiness',
    'other': 'other'
}

# Apply mapping to standardize labels
merged_df['emotion'] = merged_df['emotion'].replace(emotion_mapping)


# Step 10: Remove 'other' emotion category (optional)
before_other = len(merged_df)
merged_df = merged_df[merged_df['emotion'] != 'other']
after_other = len(merged_df)
print("Removed 'other' emotion labels:", before_other - after_other)
print("Remaining rows after removing 'other' category:", len(merged_df))


# Step 11: Check distribution of emotions in final dataset
print("\nDistribution of emotions:")
print(merged_df['emotion'].value_counts())


# Step 12: Final check
print("Final dataset size:", len(merged_df))


# Step 13: Save the cleaned dataset
merged_df.to_csv("emotion_cleaned.csv", index=False, encoding="utf-8-sig")
files.download("emotion_cleaned.csv")

