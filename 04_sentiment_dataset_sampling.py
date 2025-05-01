"""
Persian Sentiment Analysis - Balanced Dataset Creation
-----------------------------------------------------
This script creates a balanced dataset for Persian sentiment analysis by sampling an equal number of entries from each sentiment class.
Input: Cleaned CSV file with Persian Instagram comments and sentiment labels
Output: Balanced CSV with equal representation of each sentiment class
Dependencies: pandas (tested with version 2.2.2), numpy (tested with version 2.0.2)

Purpose: The balanced dataset provides an unbiased analysis.
This script takes the previously cleaned sentiment dataset and creates a balanced version
by sampling an equal number of entries from each sentiment class (positive, negative, neutral).
"""


# Step 1: Import required libraries
import pandas as pd
import numpy as np
from google.colab import files
import io


# Step 2: Upload the cleaned CSV file
print("Please upload your sentiment cleaned CSV file: (sentiment_cleaned.csv)")
uploaded = files.upload()


# Step 3: Get the filename of the uploaded file
file_name = list(uploaded.keys())[0]
print(f"File '{file_name}' uploaded successfully.")


# Step 4: Read the CSV file with appropriate encoding for Persian text
# UTF-8-SIG encoding is used to properly handle Persian characters
df = pd.read_csv(io.BytesIO(uploaded[file_name]), encoding='utf-8-sig')


# Step 5: Display basic information about the original dataset
print("\n=== ORIGINAL DATASET INFORMATION ===")
print(f"Total number of entries: {len(df)}")
print(f"Columns: {', '.join(df.columns)}")

# Show the distribution of sentiment labels
print("\nSentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{sentiment.capitalize()}: {count} entries ({percentage:.1f}%)")

# Step 6: Set a fixed random seed for reproducibility
random_seed = 42
print(f"\nUsing fixed random seed: {random_seed} for reproducibility")

# Step 7: Create a balanced dataset by sampling an equal number from each sentiment class
sample_size = 300  # Number of entries to sample from each sentiment class
sampled_data = pd.DataFrame()

# Process each sentiment category
for sentiment in ['negative', 'positive', 'neutral']:
    # Filter the dataframe for the current sentiment
    sentiment_df = df[df['sentiment'] == sentiment]
    
    # Check if we have enough data for this sentiment
    if len(sentiment_df) < sample_size:
        print(f"\nWarning: Not enough data for '{sentiment}'. Only {len(sentiment_df)} entries available.")
        # Take all available data for this sentiment
        sampled_sentiment = sentiment_df
    else:
        # Randomly sample the specified number of entries with a fixed random seed for reproducibility
        sampled_sentiment = sentiment_df.sample(n=sample_size, random_state=random_seed)
    
    # Add to the sampled dataframe
    sampled_data = pd.concat([sampled_data, sampled_sentiment])


# Step 8: Shuffle the sampled data to avoid any sequence bias
# Using a fixed random seed ensures reproducibility
sampled_data = sampled_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)


# Step 9: Display information about the balanced dataset
print("\n=== BALANCED DATASET INFORMATION ===")
print(f"Total number of entries: {len(sampled_data)}")

# Show the distribution of sentiment labels in the balanced dataset
print("\nBalanced Sentiment Distribution:")
sampled_sentiment_counts = sampled_data['sentiment'].value_counts()
for sentiment, count in sampled_sentiment_counts.items():
    percentage = (count / len(sampled_data)) * 100
    print(f"{sentiment.capitalize()}: {count} entries ({percentage:.1f}%)")


# Step 10: Save the balanced dataset to a new CSV file with proper encoding
output_file = 'sentiment_balanced.csv'
sampled_data.to_csv(output_file, index=False, encoding='utf-8-sig')


# Step 11: Download the balanced dataset
files.download(output_file)
print(f"\nFile '{output_file}' has been created and downloaded successfully.")


# Step 12: Comparison summary between original and balanced datasets
print("\n=== COMPARISON SUMMARY ===")
print(f"Original dataset size: {len(df)} entries")
print(f"Balanced dataset size: {len(sampled_data)} entries")
print(f"Reduction: {len(df) - len(sampled_data)} entries ({((len(df) - len(sampled_data)) / len(df)) * 100:.1f}%)")

# Calculate and display the change in distribution
print("\nPercentage point changes:")
for sentiment in ['negative', 'positive', 'neutral']:
    original_percentage = (sentiment_counts.get(sentiment, 0) / len(df)) * 100
    balanced_percentage = (sampled_sentiment_counts.get(sentiment, 0) / len(sampled_data)) * 100
    change = balanced_percentage - original_percentage
    print(f"{sentiment.capitalize()}: {change:+.1f} percentage points")
