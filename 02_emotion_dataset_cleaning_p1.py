"""
Persian Emotion Detection Dataset Cleaning Part 1
-------------------------------------------
This script cleans and preprocesses the Persian social media and e-commerce dataset for emotion detection.
Input: Raw TSV files with social media and e-commerce entries and emotion labels
Output: Cleaned CSV with standardized text and emotion labels
Dependencies: pandas (tested with version 2.2.2), numpy (tested with version 2.0.2), matplotlib (tested with version 3.10.0), seaborn (tested with version 0.13.2)
"""


# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
print("\n")

# View first few rows of merged dataset
merged_df.head()


# Step 5: Visualize original emotion distribution before cleaning
print("\n\n=== EMOTION DISTRIBUTION BEFORE CLEANING ===")
emotions_before = merged_df['emotion'].value_counts()
total_before = len(merged_df)

# Print statistics
print(f"Total entries: {total_before}")
for emotion, count in emotions_before.items():
    percentage = (count / total_before) * 100
    print(f"{emotion.capitalize()}: {count} entries ({percentage:.1f}%)")

# Create visualization
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='emotion', hue='emotion', data=merged_df, palette='viridis', 
                   order=emotions_before.index, legend=False)
plt.title('Emotion Distribution - Before Cleaning', fontsize=16)
plt.xlabel('Emotion', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
print("\n\n")


# Step 6: Remove empty or missing text entries
before_empty = len(merged_df)
merged_df = merged_df.dropna(subset=['text'])
merged_df = merged_df[merged_df['text'].astype(str).str.strip() != '']
after_empty = len(merged_df)
print("Removed empty texts:", before_empty - after_empty)
print("Rows after removing empty texts:", len(merged_df))
print("\n")


# Step 7: Remove duplicate text entries to avoid bias
before_duplicates = len(merged_df)
merged_df = merged_df.drop_duplicates(subset='text')
after_duplicates = len(merged_df)
print("Removed duplicate texts:", before_duplicates - after_duplicates)
print("Rows after removing duplicate texts:", len(merged_df))
print("\n")


# Step 8: Apply minimum text length threshold of 10 characters
# Very short texts typically don't contain enough information for emotion detection
before_short = len(merged_df)
merged_df = merged_df[merged_df['text'].astype(str).str.len() >= 10]
after_short = len(merged_df)
print("Removed short texts:", before_short - after_short)
print("Remaining rows after removing short texts:", len(merged_df))
print("\n")


# Step 9: Remove entries with more than 100 words
# Extremely long texts might be outliers or not typical social media content
before_long = len(merged_df)
merged_df = merged_df[merged_df['text'].apply(lambda x: len(str(x).split()) <= 100)]
after_long = len(merged_df)
print("Removed long texts:", before_long - after_long)
print("Remaining rows after removing long texts:", len(merged_df))
print("\n")


# Step 10: Standardize emotion labels for consistency
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


# Step 11: Remove 'other' emotion category (optional)
before_other = len(merged_df)
merged_df = merged_df[merged_df['emotion'] != 'other']
after_other = len(merged_df)
print("Removed 'other' emotion labels:", before_other - after_other)
print("Remaining rows after removing 'other' category:", len(merged_df))
print("\n")


# Step 12: Check distribution of emotions in final dataset
print("\nDistribution of emotions:")
print(merged_df['emotion'].value_counts())


# Step 13: Final check
print("Final dataset size:", len(merged_df))


# Step 14: Visualize the cleaned data
print("\n\n=== EMOTION DISTRIBUTION AFTER CLEANING ===")
emotions_after = merged_df['emotion'].value_counts()
total_after = len(merged_df)

# Print statistics
print(f"Total entries: {total_after}")
for emotion, count in emotions_after.items():
    percentage = (count / total_after) * 100
    print(f"{emotion.capitalize()}: {count} entries ({percentage:.1f}%)")

# Create visualization
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='emotion', hue='emotion', data=merged_df, palette='viridis', 
                   order=emotions_after.index, legend=False)
plt.title('Emotion Distribution - After Cleaning', fontsize=16)
plt.xlabel('Emotion', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
print("\n")

# Visualize the impact of cleaning on dataset size
plt.figure(figsize=(10, 6))
cleaning_steps = ['Original', 'After Empty Removal', 'After Duplicates Removal', 
                  'After Short Text Removal', 'After Long Text Removal', 'After Other Label Removal']
counts = [total_before, after_empty, after_duplicates, after_short, after_long, total_after]

plt.bar(cleaning_steps, counts, color='teal')
plt.title('Dataset Size After Each Cleaning Step', fontsize=16)
plt.xlabel('Cleaning Step', fontsize=14)
plt.ylabel('Number of Entries', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add count labels on top of each bar
for i, count in enumerate(counts):
    plt.text(i, count + 100, f'{count}', ha='center')

plt.tight_layout()
plt.show()


# Step 15: Save the cleaned dataset
merged_df.to_csv("emotion_cleaned.csv", index=False, encoding="utf-8-sig")
files.download("emotion_cleaned.csv")

