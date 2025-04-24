# Step 1: Import pandas
import pandas as pd

# Step 2: Upload the dataset
from google.colab import files
uploaded = files.upload()


# Step 3: Load the dataset using UTF-8-SIG encoding (important for Persian text)
df = pd.read_csv("dataset1.csv", encoding="utf-8-sig")

# View first few rows
df.head()


print("Number of rows:", len(df))


# Step 4: Drop the unnamed index column if it exists (e.g., 'z')
if 'z' in df.columns:
    df = df.drop(columns=['z'])

# View first few rows
df.head() 


# Step 5: Rename the 'comment' column to 'text'
df = df.rename(columns={'comment': 'text'})

df.head() 


# Step 6: Convert numerical sentiment labels to text
label_mapping = {
    1: 'positive',
    0: 'neutral',
    -1: 'negative'
}
df['sentiment'] = df['sentiment'].map(label_mapping)

df.head() 


# Step 7: Remove duplicate rows

# Drop exact duplicates based on the 'comment' column
df = df.drop_duplicates(subset='text')

# Check how many rows left
print("Rows after removing duplicates:", len(df))

# Step 8: Drop empty or missing texts (if any)

df = df.dropna(subset=['text'])
df = df[df['text'].astype(str).str.strip() != '']

print("Remaining rows after removing empty comments:", len(df))


# Step 9: Apply a minimum text length threshold of 10 characters
before_short = len(df)
df = df[df['text'].astype(str).str.len() >= 10]
after_short = len(df)
print("Removed short texts:", before_short - after_short)


# Step 10: Remove entries with more than 100 words
before_long = len(df)
df = df[df['text'].apply(lambda x: len(str(x).split()) <= 100)]
after_long = len(df)
print("Removed long texts (more than 100 words):", before_long - after_long)


print("Number of rows now:", len(df))


# Step 11: Final check
print("Final dataset size:", len(df))

# Step 12: Save the cleaned dataset
df.to_csv("dataset1_cleaned.csv", index=False, encoding="utf-8-sig")
files.download("dataset1_cleaned.csv")
