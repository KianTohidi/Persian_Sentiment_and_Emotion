"""
Persian Emotion Detection Dataset Cleaning Part 2
-------------------------------------------
This script identifies and removes e-commerce reviews from the previously cleaned Persian emotion detection dataset.
Input: Cleaned CSV from Part 1 containing text entries and emotion labels
Output: Final CSV with e-commerce reviews removed
Dependencies: pandas (tested with version 2.2.2), numpy (tested with version 2.0.2), Python math module built-in
"""


# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import math
from google.colab import files


# Step 2: Upload the previously cleaned dataset
# IMPORTANT: Before running this cell, ensure 'emotion_cleaned.csv' is available for upload
uploaded = files.upload()


# Step 3: Load the cleaned emotion dataset
df = pd.read_csv("emotion_cleaned.csv")
print(f"Total rows in dataset: {len(df)}")


# Step 4: Calculate the number of processing batches
total_batches = math.ceil(len(df) / 100)
print(f"This will be processed in {total_batches} batches of 100 texts")


# Step 5: Add a column to track e-commerce entries
if 'is_ecommerce' not in df.columns:
    df['is_ecommerce'] = False


# Step 6: Define function to extract a batch of texts with numbers
def get_batch(batch_number, batch_size=100):
    """
    Extract a batch of texts with sequential numbering for review
    
    Parameters:
    batch_number (int): The batch number to extract
    batch_size (int): Number of texts in each batch (default: 100)
    
    Returns:
    tuple: (String of numbered texts, DataFrame slice of current batch)
    """
    start_idx = (batch_number - 1) * batch_size
    end_idx = min(start_idx + batch_size, len(df))
    current_batch = df.iloc[start_idx:end_idx].copy()
    
    # Create numbered text for easy review
    numbered_texts = []
    for i, (_, row) in enumerate(current_batch.iterrows(), 1):
        numbered_texts.append(f"{i}. {row['text']}")
    
    print(f"\nBatch {batch_number} of {total_batches} (Rows {start_idx+1}-{end_idx}):")
    return '\n\n'.join(numbered_texts), current_batch


# Step 7: Define function to mark entries for removal
def mark_for_removal(batch_num, flagged_numbers):
    """
    Mark texts identified as e-commerce reviews for removal
    
    Parameters:
    batch_num (int): The batch number just reviewed
    flagged_numbers (list): List of numbers that were flagged (e.g., [3, 15, 27])
    """
    global df
    start_idx = (batch_num - 1) * 100
    
    for num in flagged_numbers:
        # Convert to 0-based indexing within the dataframe
        idx = start_idx + num - 1
        if idx < len(df):
            df.at[idx, 'is_ecommerce'] = True
            print(f"Marked text {num} for removal: {df.iloc[idx]['text'][:50]}...")
    
    print(f"Marked {len(flagged_numbers)} texts for removal in batch {batch_num}")


# Step 8: Define function to save the final dataset
def save_final_dataset():
    """
    Remove all entries marked as e-commerce and save the final dataset
    """
    # Remove all entries marked as e-commerce
    initial_count = len(df)
    final_df = df[~df['is_ecommerce']].copy()
    final_df = final_df.drop(columns=['is_ecommerce'])
    removed_count = initial_count - len(final_df)
    
    # Save the cleaned dataset
    final_df.to_csv("emotion_ecommerce_cleaned.csv", index=False, encoding="utf-8-sig")
    print(f"\nFinal cleaning complete!")
    print(f"Removed {removed_count} e-commerce reviews")
    print(f"Final dataset contains {len(final_df)} texts")
    print(f"Saved as: emotion_ecommerce_cleaned.csv")
    
    # Show distribution of emotions in final dataset
    print("\nEmotion distribution in final dataset:")
    print(final_df['emotion'].value_counts())


# Step 9: Example usage - Process first batch of texts
# Note: Change batch_num to process different batches
batch_num = 1
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nReview everything below this line:")
print("---------------------------------------------------")
print(numbered_text)



# After review, use mark_for_removal() to flag e-commerce texts

# Example: mark_for_removal(1, [3, 7, 12]) Replace with your actual batch number and text numbers

# When all batches have been reviewed, call save_final_dataset() to finalize



save_final_dataset()

from google.colab import files
files.download("dataset3_final_cleaned.csv")













