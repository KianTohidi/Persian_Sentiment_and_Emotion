"""
Persian Sentiment Analysis Dataset Cleaning
-------------------------------------------
This script cleans and preprocesses the Persian Instagram comments dataset for sentiment analysis.
Input: Raw CSV file with Instagram comments and sentiment labels
Output: Cleaned CSV with standardized text and sentiment labels
Dependencies: pandas (tested with version 2.2.2), numpy (tested with version 2.0.2), matplotlib (tested with version 3.10.0), seaborn (tested with version 0.13.2)
"""


# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Step 2: Upload the dataset
# IMPORTANT: Before running this cell:
# 1. Download the CSV dataset from the Kaggle website (link in the README)
# 2. Extract the zip file
# 3. Rename the file from "Instagram labeled comments.csv" to "sentiment.csv"
from google.colab import files
uploaded = files.upload()


# Step 3: Load the dataset using UTF-8-SIG encoding (important for Persian text)
df = pd.read_csv("sentiment.csv", encoding="utf-8-sig")

print("\n")
print("Number of rows:", len(df))
print("\n")

# View first few rows
df.head()


# Step 4: Drop the unnamed index column if it exists
unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)
    
# View first few rows
df.head() 


# Step 5: Rename the 'comment' column to 'text' for consistency
df = df.rename(columns={'comment': 'text'})

df.head() 


# Step 6: Convert numerical sentiment labels to text for better readability
label_mapping = {
    1: 'positive',
    0: 'neutral',
    -1: 'negative'
}
df['sentiment'] = df['sentiment'].map(label_mapping)

df.head() 


# Step 7: Visualize sentiment distribution BEFORE cleaning
plt.figure(figsize=(10, 6))
sentiment_before = df['sentiment'].value_counts()

# Use hue parameter
ax = sns.countplot(x='sentiment', hue='sentiment', data=df, palette='viridis', legend=False)
plt.title('Sentiment Distribution - Before Cleaning', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

# Calculate percentages
total_before = len(df)
print("\n\n=== SENTIMENT DISTRIBUTION BEFORE CLEANING ===")
print(f"Total entries: {total_before}")
for sentiment, count in sentiment_before.items():
    percentage = (count / total_before) * 100
    print(f"{sentiment.capitalize()}: {count} entries ({percentage:.1f}%)")

plt.tight_layout()
plt.show()
print("\n\n")


# Step 8: Remove duplicate rows to avoid bias in analysis
# Drop exact duplicates based on the 'text' column

before_duplicates = len(df)
df = df.drop_duplicates(subset='text')
after_duplicates = len(df)

print("Removed duplicate rows:", before_duplicates - after_duplicates)
print("Rows after removing duplicate texts:", after_duplicates)
print("\n")


# Step 9: Drop empty or missing texts (if any)
before_empty = len(df)
df = df.dropna(subset=['text'])
df = df[df['text'].astype(str).str.strip() != '']
after_empty = len(df)

print("Removed empty rows:", before_empty - after_empty)
print("Remaining rows after removing empty rows:", after_empty)
print("\n")


# Step 10: Apply a minimum text length threshold of 10 characters
# Very short texts typically don't contain enough information for sentiment analysis
before_short = len(df)
df = df[df['text'].astype(str).str.len() >= 10]
after_short = len(df)

print("Removed short texts:", before_short - after_short)
print("Remaining rows after removing short texts:", len(df))
print("\n")


# Step 11: Remove entries with more than 100 words
# Extremely long texts might be outliers or not typical social media comments
before_long = len(df)
df = df[df['text'].apply(lambda x: len(str(x).split()) <= 100)]
after_long = len(df)

print("Removed long texts:", before_long - after_long)
print("Remaining rows after removing long texts:", len(df))
print("\n")


# Step 12: Final check
print("\n=== CLEANING SUMMARY ===")
print(f"Starting dataset size: {total_before} entries")
print(f"Final dataset size: {len(df)} entries")
print(f"Total removed: {total_before - len(df)} entries ({((total_before - len(df)) / total_before) * 100:.1f}%)")
print("\n\n")


# Step 13: Visualize sentiment distribution AFTER cleaning
plt.figure(figsize=(10, 6))
sentiment_after = df['sentiment'].value_counts()

# Use hue parameter instead of palette to avoid FutureWarning
ax = sns.countplot(x='sentiment', hue='sentiment', data=df, palette='viridis', legend=False)
plt.title('Sentiment Distribution - After Cleaning', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

# Calculate percentages
total_after = len(df)
print("\n\n=== SENTIMENT DISTRIBUTION AFTER CLEANING ===")
print(f"Total entries: {total_after}")
for sentiment, count in sentiment_after.items():
    percentage = (count / total_after) * 100
    print(f"{sentiment.capitalize()}: {count} entries ({percentage:.1f}%)")

plt.tight_layout()
plt.show()
print("\n\n")


# Step 14: Compare distributions before and after cleaning
print("\n=== DISTRIBUTION CHANGE ANALYSIS ===")
print(f"Total reduction: {total_before - total_after} entries ({((total_before - total_after) / total_before) * 100:.1f}%)")

# Create a side-by-side comparison bar chart
plt.figure(figsize=(12, 7))

# Create a simpler approach with a standard bar chart
sentiments = list(label_mapping.values())
before_percentages = [(sentiment_before.get(s, 0) / total_before) * 100 for s in sentiments]
after_percentages = [(sentiment_after.get(s, 0) / total_after) * 100 for s in sentiments]

# Set up the bar chart positions
x = np.arange(len(sentiments))
width = 0.35

# Create the bars
fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, before_percentages, width, label='Before Cleaning (%)')
rects2 = ax.bar(x + width/2, after_percentages, width, label='After Cleaning (%)')

# Add labels and title
ax.set_xlabel('Sentiment', fontsize=14)
ax.set_ylabel('Percentage', fontsize=14)
ax.set_title('Sentiment Distribution - Before vs After Cleaning (%)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(sentiments)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Calculate and display the change in distribution
print("\nPercentage point changes:")
for i, sentiment in enumerate(sentiments):
    change = after_percentages[i] - before_percentages[i]
    print(f"{sentiment.capitalize()}: {change:+.1f} percentage points")

plt.tight_layout()
plt.show()
print("\n\n")


# Step 15: Save the cleaned dataset
df.to_csv("sentiment_cleaned.csv", index=False, encoding="utf-8-sig")
files.download("sentiment_cleaned.csv")

