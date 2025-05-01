"""
Persian Emotion Detection - Balanced Dataset Creation
----------------------------------------------------
This script creates a balanced dataset for Persian emotion detection by sampling an equal number of entries from each emotion class.
Input: Cleaned CSV file with Persian texts and emotion labels (emotion_ecommerce_cleaned.csv)
Output: Balanced CSV with equal representation of each emotion class (emotion_balanced.csv)
Dependencies: pandas (tested with version 2.2.2), numpy (tested with version 2.0.2)

Purpose: The balanced dataset provides an unbiased analysis.
This script takes the previously cleaned emotion dataset and creates a balanced version
by sampling an equal number of entries from each emotion class (sadness, happiness, surprise, fear, hate, anger).
"""


# Step 1: Import required libraries
import pandas as pd
import numpy as np
from google.colab import files
import io


# Step 2: Upload the cleaned CSV file
print("Please upload your emotion ecommerce cleaned CSV file: (emotion_ecommerce_cleaned.csv)")
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

# Show the distribution of emotion labels
print("\nEmotion Distribution:")
emotion_counts = df['emotion'].value_counts()
for emotion, count in emotion_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{emotion.capitalize()}: {count} entries ({percentage:.1f}%)")


# Step 6: Set a fixed random seed for reproducibility
random_seed = 42
print(f"\nUsing fixed random seed: {random_seed} for reproducibility")


# Step 7: Create a balanced dataset by sampling an equal number from each emotion class
sample_size = 300  # Number of entries to sample from each emotion class
sampled_data = pd.DataFrame()

# List of emotions to sample
emotions = ['sadness', 'happiness', 'surprise', 'fear', 'hate', 'anger']

# Process each emotion category
for emotion in emotions:
    # Filter the dataframe for the current emotion
    emotion_df = df[df['emotion'] == emotion]
    
    # Check if we have enough data for this emotion
    if len(emotion_df) < sample_size:
        print(f"\nWarning: Not enough data for '{emotion}'. Only {len(emotion_df)} entries available.")
        # Take all available data for this emotion
        sampled_emotion = emotion_df
    else:
        # Randomly sample the specified number of entries with a fixed random seed for reproducibility
        sampled_emotion = emotion_df.sample(n=sample_size, random_state=random_seed)
    
    # Add to the sampled dataframe
    sampled_data = pd.concat([sampled_data, sampled_emotion])


# Step 8: Shuffle the sampled data to avoid any sequence bias
# Using a fixed random seed ensures reproducibility
sampled_data = sampled_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)


# Step 9: Display information about the balanced dataset
print("\n=== BALANCED DATASET INFORMATION ===")
print(f"Total number of entries: {len(sampled_data)}")

# Show the distribution of emotion labels in the balanced dataset
print("\nBalanced Emotion Distribution:")
sampled_emotion_counts = sampled_data['emotion'].value_counts()
for emotion, count in sampled_emotion_counts.items():
    percentage = (count / len(sampled_data)) * 100
    print(f"{emotion.capitalize()}: {count} entries ({percentage:.1f}%)")


# Step 10: Save the balanced dataset to a new CSV file with proper encoding
output_file = 'emotion_balanced.csv'
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
for emotion in emotions:
    original_percentage = (emotion_counts.get(emotion, 0) / len(df)) * 100
    balanced_percentage = (sampled_emotion_counts.get(emotion, 0) / len(sampled_data)) * 100
    change = balanced_percentage - original_percentage
    print(f"{emotion.capitalize()}: {change:+.1f} percentage points")

