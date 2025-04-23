# Import necessary libraries
import pandas as pd
import numpy as np
import math
from google.colab import files
uploaded = files.upload() 
# Load your cleaned dataset
df = pd.read_csv("dataset3_cleaned.csv")
print(f"Total rows in dataset: {len(df)}")
# Calculate how many batches of 100 we'll have
total_batches = math.ceil(len(df) / 100)
print(f"This will be processed in {total_batches} batches of 100 texts")
# Create a column to track whether entries should be removed
if 'is_ecommerce' not in df.columns:
    df['is_ecommerce'] = False




# Function to extract a batch of texts with numbers
def get_batch(batch_number, batch_size=100):
    start_idx = (batch_number - 1) * batch_size
    end_idx = min(start_idx + batch_size, len(df))
    current_batch = df.iloc[start_idx:end_idx].copy()
    
    # Create numbered text for easy review
    numbered_texts = []
    for i, (_, row) in enumerate(current_batch.iterrows(), 1):
        numbered_texts.append(f"{i}. {row['text']}")
    
    print(f"\nBatch {batch_number} of {total_batches} (Rows {start_idx+1}-{end_idx}):")
    return '\n\n'.join(numbered_texts), current_batch

# Function to mark entries for removal
def mark_for_removal(batch_num, flagged_numbers):
    """
    Mark texts identified as e-commerce reviews for removal
    
    Parameters:
    batch_num (int): The batch number you just reviewed
    flagged_numbers (list): List of numbers that Claude flagged (e.g., [3, 15, 27])
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

# Function to save the final dataset
def save_final_dataset():
    # Remove all entries marked as e-commerce
    initial_count = len(df)
    final_df = df[~df['is_ecommerce']].copy()
    final_df = final_df.drop(columns=['is_ecommerce'])
    removed_count = initial_count - len(final_df)
    
    # Save the cleaned dataset
    final_df.to_csv("dataset3_final_cleaned.csv", index=False, encoding="utf-8-sig")
    print(f"\nFinal cleaning complete!")
    print(f"Removed {removed_count} e-commerce reviews")
    print(f"Final dataset contains {len(final_df)} texts")
    print(f"Saved as: dataset3_final_cleaned.csv")
    
    # Show distribution of emotions in final dataset
    print("\nEmotion distribution in final dataset:")
    print(final_df['emotion'].value_counts())





# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 1  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(1, [27, 31, 41, 56, 84, 97])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 2  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(2, [33, 34, 42, 55, 64, 58, 82])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 3  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(3, [37, 44, 51, 72, 78])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 4  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(4, [41, 69, 89, 91])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 5  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(5, [12, 50, 64, 95])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 6  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(6, [27, 36, 88, 98])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 7  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(7, [21, 31, 44, 53, 55, 58, 66])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 8  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)

mark_for_removal(8, [36, 46, 47, 49, 98])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 9  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(9, [32, 37, 42, 43, 70, 93])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 10  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(10, [17, 18, 41, 48, 53, 58, 59, 61, 65, 67, 78, 96, 100])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 11  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(11, [12, 32, 80, 81, 86, 90])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 12  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)





mark_for_removal(12, [27, 29, 71, 90])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 13  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(13, [3, 16, 38, 44, 50, 62, 78, 89])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 14  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(14, [26, 48, 57, 65, 72, 78, 83])  # Replace with your actual batch number and text numbers




# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 15  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(15, [13, 43, 46, 57, 68, 72, 84])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 16  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(16, [2, 9, 27, 28, 34, 73, 84])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 17  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)




mark_for_removal(17, [8, 43, 45, 47, 55, 68, 73])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 18  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(18, [7, 35, 37, 46, 50, 51, 82, 92])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 19  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(19, [6, 37, 56, 57, 60, 64, 79, 85, 86, 94, 95])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 20  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(20, [26, 33, 41, 53, 61, 87])  # Replace with your actual batch number and text numbers
# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 21  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(21, [12, 13, 24, 35, 40, 53, 67, 83])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 22  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)

mark_for_removal(22, [16, 26, 43, 55, 69, 76, 85, 90, 91, 97, 99])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 23  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(23, [18, 29, 33, 62, 83, 85, 89])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 24  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(24, [21, 79, 89, 90, 98])  # Replace with your actual batch number and text numbers



# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 25  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)

mark_for_removal(25, [64])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 26  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)

mark_for_removal(26, [4, 5, 17, 31, 34, 46, 82, 88, 89, 93])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 27  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(27, [11, 19, 23, 26, 34, 39, 55, 65, 67, 88, 92])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 28  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(28, [6, 10, 11, 13, 17, 24, 36, 65, 67, 74, 79, 97])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 29  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)

mark_for_removal(29, [24, 27, 48, 54, 75, 77, 99])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 30  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(30, [24, 25, 46, 62, 81, 85, 86, 96])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 31  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(31, [13, 47, 49, 51, 67, 74, 83, 89, 93, 100])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 32  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(32, [7, 8, 17, 23, 27, 32, 41, 64, 72, 73, 100])  # Replace with your actual batch number and text numbers



# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 33  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(33, [8, 27, 33, 34, 35, 37, 48, 65, 81, 91, 94])  # Replace with your actual batch number and text numbers



# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 34  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(34, [18, 27, 34, 42, 46, 47, 63, 64, 71, 79, 85, 88])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 35  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(35, [4, 8, 24, 39])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 36  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(36, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 37  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(37, [])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 38  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(38, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 39  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(39, [])  # Replace with your actual batch number and text numbers



# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 40  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(40, [])  # Replace with your actual batch number and text numbers



# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 41  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(41, [])  # Replace with your actual batch number and text numbers
# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 42  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(42, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 43  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(43, [])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 44  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(44, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 45  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(45, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 46  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(46, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 47  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(47, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 48  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(48, [])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 49  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(49, [])  # Replace with your actual batch number and text numbers

# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 50  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)


mark_for_removal(50, [])  # Replace with your actual batch number and text numbers



# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 51  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(51, [])  # Replace with your actual batch number and text numbers



# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 52  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(52, [86])  # Replace with your actual batch number and text numbers


# Example usage - run this to get the first batch
# Comment out when not needed
batch_num = 53  # Change this number to get different batches
numbered_text, current_batch = get_batch(batch_num)
print(f"Generated {len(current_batch)} numbered texts for review")
print("\nCopy everything below this line to send to Claude:")
print("---------------------------------------------------")
print(numbered_text)



mark_for_removal(53, [])  # Replace with your actual batch number and text numbers




save_final_dataset()


# After running save_final_dataset()
from google.colab import files
files.download("dataset3_final_cleaned.csv")













