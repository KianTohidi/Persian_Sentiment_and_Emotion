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
# Due to repetition, I just shared one below. For visual analysis, all 54 batches need to be done

batch_num = 1
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nReview everything below this line:")
print("---------------------------------------------------")
print(numbered_text)


# Step 10: After review, use mark_for_removal() to flag e-commerce texts
# Example: mark_for_removal(1, [3, 7, 12]) Replace with your actual batch number and text numbers
# Below, you can see the removal of e-commerce entries that had been conducted. Shared for reproducibility 

print("\n\n")
mark_for_removal(1, [1, 3, 7, 9, 20, 38, 42, 54, 74, 84, 97])
mark_for_removal(2, [22, 23, 26, 28, 32, 52, 69, 71])
mark_for_removal(3, [13, 32, 45, 58, 80, 86, 92])
mark_for_removal(4, [4, 20, 31, 43, 68, 90, 99])
mark_for_removal(5, [6, 7, 14, 20, 25, 36, 55, 85, 88, 99])
mark_for_removal(6, [10, 14, 26, 39, 44, 51, 69, 70, 76, 86])
mark_for_removal(7, [15, 26, 50, 85, 87, 89, 97])
mark_for_removal(8, [10, 15, 50, 57, 78, 80, 89, 93, 94])
mark_for_removal(9, [26, 36, 50, 59, 82])
mark_for_removal(10, [1, 2, 5, 9, 24, 30, 31, 39, 40, 71, 78, 86, 98])
mark_for_removal(11, [6, 32, 57, 58, 69, 80, 85, 98])
mark_for_removal(12, [12, 28, 41, 59, 61, 71, 88, 100])
mark_for_removal(13, [14, 21, 24, 30, 35, 36, 42, 44, 63, 74, 78, 83])
mark_for_removal(14, [7, 28, 30, 34, 66])
mark_for_removal(15, [24, 34, 35, 43])
mark_for_removal(16, [9, 49, 50, 61, 62, 76, 79, 91])
mark_for_removal(17, [27, 33, 34, 38, 56, 64, 68, 71, 79, 84, 100])
mark_for_removal(18, [10, 12, 33, 37, 51, 55, 56, 58, 62, 65, 69, 81])
mark_for_removal(19, [10, 12, 19, 24, 42, 69, 72, 93, 99])
mark_for_removal(20, [21, 23, 45, 70, 71, 92])
mark_for_removal(21, [8, 27, 31, 32, 42, 59, 93, 95, 97])
mark_for_removal(22, [13, 20, 29, 39, 46, 53, 54, 63, 69, 73, 78, 87])
mark_for_removal(23, [10, 18, 19, 46, 54, 73, 79, 80, 81, 83, 94])
mark_for_removal(24, [11, 27, 37, 40, 64, 73, 80, 88, 92, 93])
mark_for_removal(25, [9, 10, 16, 18, 26, 32, 35, 51, 55, 71, 86])
mark_for_removal(26, [])
mark_for_removal(27, [])
mark_for_removal(28, [])
mark_for_removal(29, [])
mark_for_removal(30, [])
mark_for_removal(31, [])
mark_for_removal(32, [])
mark_for_removal(33, [])
mark_for_removal(34, [])
mark_for_removal(35, [])
mark_for_removal(36, [])
mark_for_removal(37, [])
mark_for_removal(38, [])
mark_for_removal(39, [])
mark_for_removal(40, [])
mark_for_removal(41, [])
mark_for_removal(42, [])
mark_for_removal(43, [70])
mark_for_removal(44, [74, 78, 88])
mark_for_removal(45, [3, 44, 80, 81, 89])
mark_for_removal(46, [2, 5, 11, 29, 84, 91, 98])
mark_for_removal(47, [19, 25, 88])
mark_for_removal(48, [16, 36, 38, 59, 97])
mark_for_removal(49, [41, 73, 82])
mark_for_removal(50, [34, 44, 67, 77, 90, 99])
mark_for_removal(51, [1, 4, 81, 91, 92, 94])
mark_for_removal(52, [42, 76, 81, 86, 87])
mark_for_removal(53, [14, 37, 61, 62, 85, 92, 97])
mark_for_removal(54, [2])


# Step 11: When all batches have been reviewed, call save_final_dataset() to finalize

save_final_dataset()

from google.colab import files
files.download("emotion_ecommerce_cleaned.csv")

