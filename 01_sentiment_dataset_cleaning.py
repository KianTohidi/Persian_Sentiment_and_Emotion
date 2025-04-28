"""
Persian Sentiment Analysis Dataset Cleaning
-------------------------------------------
This script cleans and preprocesses the Persian Instagram comments dataset for sentiment analysis.
Input: Raw CSV file with Instagram comments and sentiment labels
Output: Cleaned CSV with standardized text and sentiment labels
Dependencies: pandas (tested with version 2.2.2)
"""


# Step 1: Import pandas
import pandas as pd


# Step 2: Upload the dataset
# IMPORTANT: Before running this cell:
# 1. Download the CSV dataset from the Kaggle website (link in the README)
# 2. Extract the zip file
# 3. Rename the file from "Instagram labeled comments.csv" to "sentiment.csv"
from google.colab import files
uploaded = files.upload()


# Step 3: Load the dataset using UTF-8-SIG encoding (important for Persian text)
df = pd.read_csv("sentiment.csv", encoding="utf-8-sig")

# View first few rows
df.head()

print("Number of rows:", len(df))


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


# Step 7: Remove duplicate rows to avoid bias in analysis
# Drop exact duplicates based on the 'text' column
df = df.drop_duplicates(subset='text')

# Check how many rows left
print("Rows after removing duplicates:", len(df))


# Step 8: Drop empty or missing texts (if any)
df = df.dropna(subset=['text'])
df = df[df['text'].astype(str).str.strip() != '']

print("Remaining rows after removing empty comments:", len(df))


# Step 9: Apply a minimum text length threshold of 10 characters
# Very short texts typically don't contain enough information for sentiment analysis
before_short = len(df)
df = df[df['text'].astype(str).str.len() >= 10]
after_short = len(df)
print("Removed short texts:", before_short - after_short)
print("Remaining rows after removing short comments:", len(df))


# Step 10: Remove entries with more than 100 words
# Extremely long texts might be outliers or not typical social media comments
before_long = len(df)
df = df[df['text'].apply(lambda x: len(str(x).split()) <= 100)]
after_long = len(df)
print("Removed long texts:", before_long - after_long)
print("Remaining rows after removing long comments:", len(df))


# Step 11: Final check
print("Final dataset size:", len(df))


# Step 12: Save the cleaned dataset
df.to_csv("sentiment_cleaned.csv", index=False, encoding="utf-8-sig")
files.download("sentiment_cleaned.csv")


