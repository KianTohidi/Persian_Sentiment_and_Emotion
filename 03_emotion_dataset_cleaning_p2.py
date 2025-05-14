"""
Persian Emotion Detection Dataset Cleaning Part 2
-------------------------------------------------

This script identifies and removes e-commerce reviews from the previously cleaned Persian
emotion detection dataset to improve dataset quality for emotion analysis.

Purpose:
    Process the cleaned emotion dataset from Part 1 to identify and remove e-commerce reviews
    that could skew the emotion analysis results. The script provides visualization tools to
    compare data distributions before and after cleaning.

### IMPORTANT: Before running this code:
    1. Ensure 'emotion_cleaned.csv' from Part 1 is available for upload
    2. Run the script in batches as instructed below to review and mark e-commerce texts

Input:
    Cleaned CSV from Part 1 containing text entries and emotion labels (emotion_cleaned.csv)
    Expected format: CSV with 'text' and 'emotion' columns

Output:
    - "emotion_ecommerce_cleaned.csv": Final dataset with e-commerce reviews removed
    - Visualization reports showing emotion distribution before and after cleaning
    - Detailed statistics on data transformation during the cleaning process

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - matplotlib (tested with version 3.10.0)
    - seaborn (tested with version 0.13.2)
    - google.colab (for file upload/download in Colab environment)
"""

import logging
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from google.colab import files

# Configure logging for detailed process tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("ecommerce_cleaning.log")  # File output
    ]
)
logger = logging.getLogger(__name__)

# Constants for consistent reference
INPUT_FILE = "emotion_cleaned.csv"
OUTPUT_FILE = "emotion_ecommerce_cleaned.csv"
ENCODING = "utf-8-sig"  # Standard encoding for Persian text
BATCH_SIZE = 100       # Number of texts to process in each batch

# Dictionary containing batch numbers and corresponding e-commerce text numbers
# Pre-identified e-commerce entries for reproducibility
ECOMMERCE_BATCHES = {
    1: [1, 3, 7, 9, 20, 38, 42, 54, 74, 84, 97],
    2: [22, 23, 26, 28, 32, 52, 69, 71],
    3: [13, 32, 45, 58, 80, 86, 92],
    4: [4, 20, 31, 43, 68, 90, 99],
    5: [6, 7, 14, 20, 25, 36, 55, 85, 88, 99],
    6: [10, 14, 26, 39, 44, 51, 69, 70, 76, 86],
    7: [15, 26, 50, 85, 87, 89, 97],
    8: [10, 15, 50, 57, 78, 80, 89, 93, 94],
    9: [26, 36, 50, 59, 82],
    10: [1, 2, 5, 9, 24, 30, 31, 39, 40, 71, 78, 86, 98],
    11: [6, 32, 57, 58, 69, 80, 85, 98],
    12: [12, 28, 41, 59, 61, 71, 88, 100],
    13: [14, 21, 24, 30, 35, 36, 42, 44, 63, 74, 78, 83],
    14: [7, 28, 30, 34, 66],
    15: [24, 34, 35, 43],
    16: [9, 49, 50, 61, 62, 76, 79, 91],
    17: [27, 33, 34, 38, 56, 64, 68, 71, 79, 84, 100],
    18: [10, 12, 33, 37, 51, 55, 56, 58, 62, 65, 69, 81],
    19: [10, 12, 19, 24, 42, 69, 72, 93, 99],
    20: [21, 23, 45, 70, 71, 92],
    21: [8, 27, 31, 32, 42, 59, 93, 95, 97],
    22: [13, 20, 29, 39, 46, 53, 54, 63, 69, 73, 78, 87],
    23: [10, 18, 19, 46, 54, 73, 79, 80, 81, 83, 94],
    24: [11, 27, 37, 40, 64, 73, 80, 88, 92, 93],
    25: [9, 10, 16, 18, 26, 32, 35, 51, 55, 71, 86],
    # Batches 26-42 have no e-commerce entries
    43: [70],
    44: [74, 78, 88],
    45: [3, 44, 80, 81, 89],
    46: [2, 5, 11, 29, 84, 91, 98],
    47: [19, 25, 88],
    48: [16, 36, 38, 59, 97],
    49: [41, 73, 82],
    50: [34, 44, 67, 77, 90, 99],
    51: [1, 4, 81, 91, 92, 94],
    52: [42, 76, 81, 86, 87],
    53: [14, 37, 61, 62, 85, 92, 97],
    54: [2]
}

# Visualization settings for consistency
VIZ_SETTINGS = {
    'figsize_large': (14, 8),
    'figsize_medium': (12, 6),
    'palette': 'viridis',
    'title_fontsize': 18,
    'label_fontsize': 14,
    'tick_fontsize': 12,
    'grid_alpha': 0.3
}


def load_dataset(file_path: str = INPUT_FILE) -> Optional[pd.DataFrame]:
    """
    Load the previously cleaned emotion dataset.
    
    Purpose:
        Handles file upload in Colab environment and loads the dataset with the
        appropriate encoding for Persian text.
        
    Input:
        file_path: Path to the CSV file containing the dataset
        
    Returns:
        pd.DataFrame or None: Loaded dataset if successful, None otherwise
    """
    
    logger.info("Starting dataset load process")
    try:
        # Handle file upload in Colab
        logger.info(f"Prompting user to upload {file_path}")
        print(f"Please upload the previously cleaned dataset file from Part 1 (emotion_cleaned.csv):")
        uploaded = files.upload()
        
        if not uploaded:
            logger.warning("No file was uploaded")
            return None
        
        uploaded_files = list(uploaded.keys())
        print(f"\nUploaded file: {', '.join(uploaded_files)}")
        
        # Find matching file (handle Colab's automatic renaming)
        file_base = file_path.split('.')[0]
        file_ext = file_path.split('.')[-1]
        matching_files = [f for f in uploaded_files if 
                         (file_base in f and f.endswith(file_ext))]
        
        if matching_files:
            matched_file = matching_files[0]  # Use the first match
            logger.info(f"Found matching file '{matched_file}' for expected file '{file_path}'")
            
            try:
                logger.info(f"Loading dataset from {matched_file} with {ENCODING} encoding")
                df = pd.read_csv(matched_file, encoding=ENCODING)
                
                # Validate required columns exist
                required_columns = ['text', 'emotion']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns: {', '.join(missing_columns)}")
                    print(f"Error: The uploaded file is missing required columns: {', '.join(missing_columns)}")
                    return None
                
                logger.info(f"Successfully loaded {matched_file}: {len(df)} rows")
                print(f"Loaded {matched_file}: {len(df)} rows, {len(df.columns)} columns")
                return df
                
            except Exception as e:
                logger.error(f"Error loading {matched_file}: {str(e)}")
                print(f"Error: Failed to load {matched_file} - {str(e)}")
                return None
        else:
            logger.warning(f"No matching file found for {file_path}")
            print(f"Warning: No matching file found for {file_path}")
            return None
        
    except Exception as e:
        logger.error(f"Error in load_dataset: {str(e)}")
        print(f"Error: Failed to load dataset - {str(e)}")
        return None


def initialize_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Initialize the dataset for e-commerce detection processing.
    
    Purpose:
        Prepares the dataset by adding necessary tracking columns and
        calculating processing information.
        
    Input:
        df: DataFrame to initialize
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Initialized DataFrame with tracking columns
        - int: Total number of batches required for processing
    """
    
    logger.info("Initializing dataset for e-commerce detection")
    
    # Make a copy to avoid modifying the input DataFrame
    df_init = df.copy()
    
    # Add tracking column if it doesn't exist
    if 'is_ecommerce' not in df_init.columns:
        logger.info("Adding 'is_ecommerce' tracking column")
        df_init['is_ecommerce'] = False
    
    # Calculate processing batches
    total_batches = math.ceil(len(df_init) / BATCH_SIZE)
    logger.info(f"Dataset will be processed in {total_batches} batches of {BATCH_SIZE} texts each")
    print(f"Dataset initialization complete:")
    print(f"  - Total entries: {len(df_init)}")
    print(f"  - Processing batches: {total_batches}")
    print(f"  - Batch size: {BATCH_SIZE} texts per batch")
    
    return df_init, total_batches


def get_batch(df: pd.DataFrame, batch_number: int, batch_size: int = BATCH_SIZE) -> Tuple[str, pd.DataFrame]:
    """
    Extract a batch of texts with sequential numbering for review.
    
    Purpose:
        Creates a readable, numbered list of texts from the current batch for
        human review to identify e-commerce entries.
        
    Input:
        df: DataFrame containing the texts
        batch_number: The batch number to extract
        batch_size: Number of texts in each batch
        
    Returns:
        Tuple containing:
        - str: String of numbered texts for human review
        - pd.DataFrame: DataFrame slice of current batch
    """
    
    logger.info(f"Extracting batch {batch_number} for review")
    
    # Calculate indices for the current batch
    start_idx = (batch_number - 1) * batch_size
    end_idx = min(start_idx + batch_size, len(df))
    current_batch = df.iloc[start_idx:end_idx].copy()
    
    # Create numbered text for easy review
    numbered_texts = []
    for i, (_, row) in enumerate(current_batch.iterrows(), 1):
        # Add index number and text with proper formatting
        numbered_texts.append(f"{i}. {row['text']}")
    
    # Log batch information
    batch_text = '\n\n'.join(numbered_texts)
    logger.info(f"Generated {len(current_batch)} numbered texts for batch {batch_number}")
    print(f"\nBatch {batch_number} Ready for Review")
    print(f"Entries {start_idx+1}-{end_idx} of {len(df)} total")
    print(f"Contains {len(current_batch)} texts for review")
    
    return batch_text, current_batch


def mark_for_removal(df: pd.DataFrame, batch_num: int, flagged_numbers: List[int]) -> pd.DataFrame:
    """
    Mark texts identified as e-commerce reviews for removal.
    
    Purpose:
        Updates the DataFrame to flag specified entries as e-commerce reviews
        based on their position in the current batch.
        
    Input:
        df: DataFrame to update
        batch_num: The batch number being processed
        flagged_numbers: List of numbers within the batch that were flagged
        
    Returns:
        pd.DataFrame: Updated DataFrame with flagged entries
    """
    
    logger.info(f"Marking batch {batch_num} texts for removal: {flagged_numbers}")
    
    # Make a copy to avoid modifying the input DataFrame
    df_updated = df.copy()
    
    # Calculate starting index for this batch
    start_idx = (batch_num - 1) * BATCH_SIZE
    flagged_count = 0
    
    # Mark each flagged entry
    for num in flagged_numbers:
        # Convert to 0-based indexing within the dataframe
        idx = start_idx + num - 1
        if 0 <= idx < len(df_updated):
            df_updated.at[idx, 'is_ecommerce'] = True
            flagged_count += 1
            
            # Log the first 50 characters of each marked text
            text_preview = df_updated.iloc[idx]['text'][:50] + "..." if len(df_updated.iloc[idx]['text']) > 50 else df_updated.iloc[idx]['text']
            logger.info(f"Marked text {num} (index {idx}) as e-commerce: {text_preview}")
    
    # Report results
    logger.info(f"Successfully marked {flagged_count} texts in batch {batch_num}")
    print(f"Processed batch {batch_num}:")
    print(f"  - Marked {flagged_count} texts as e-commerce")
    print(f"  - These texts will be removed in the final dataset")
    
    return df_updated


def display_ecommerce_examples(df: pd.DataFrame, num_examples: int = 5) -> None:
    """
    Display examples of e-commerce texts that will be removed.
    
    Purpose:
        Shows researchers some examples of the e-commerce texts being
        removed to help them understand the cleaning process.
        
    Input:
        df: DataFrame with marked e-commerce entries
        num_examples: Number of example texts to display
    """
    
    logger.info(f"Preparing to display {num_examples} examples of e-commerce texts")
    
    # Get all texts marked as e-commerce
    ecommerce_texts = df[df['is_ecommerce']].copy()
    
    # If no e-commerce texts found, exit
    if len(ecommerce_texts) == 0:
        print("No e-commerce texts found to display.")
        return
    
    # Sample texts to display as examples
    sample_size = min(num_examples, len(ecommerce_texts))
    sampled_texts = ecommerce_texts.sample(sample_size)
    
    print("\n----------------------------------------")
    print("EXAMPLES OF E-COMMERCE TEXTS (TO BE REMOVED)")
    print("----------------------------------------")
    print("These are pre-identified e-commerce entries for reproducibility:\n")
    
    # Display each example with index
    for i, (idx, row) in enumerate(sampled_texts.iterrows(), 1):
        emotion = row['emotion']
        text = row['text']
        # Limit text length for display
        if len(text) > 100:
            text = text[:100] + "..."
        
        print(f"Example {i} (Original Index: {idx}, Emotion: {emotion}):")
        print(f"{text}")
        print("----------------------------------------")


def process_all_batches(df: pd.DataFrame, batch_data: Dict[int, List[int]]) -> Tuple[pd.DataFrame, int]:
    """
    Process all batches using pre-identified e-commerce entries.
    
    Purpose:
        Automates the processing of all batches using the pre-identified
        e-commerce entries dictionary for reproducibility.
        
    Input:
        df: DataFrame to process
        batch_data: Dictionary mapping batch numbers to lists of e-commerce entry numbers
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Updated DataFrame with all flagged entries
        - int: Total number of entries marked for removal
    """
    
    logger.info("Starting batch processing with pre-identified e-commerce entries")
    print("\nProcessing all batches with pre-identified e-commerce entries:")
    print("(Using pre-identified entries for reproducible results)")
    
    # Make a copy to avoid modifying the input DataFrame
    df_processed = df.copy()
    total_marked = 0
    processed_batches = 0
    
    # Process each batch with existing data
    for batch_num, text_numbers in batch_data.items():
        df_processed = mark_for_removal(df_processed, batch_num, text_numbers)
        total_marked += len(text_numbers)
        processed_batches += 1
    
    # Calculate batches with no removals
    all_batches = set(range(1, math.ceil(len(df) / BATCH_SIZE) + 1))
    processed_set = set(batch_data.keys())
    no_removal_batches = all_batches - processed_set
    
    # Display examples of e-commerce texts
    display_ecommerce_examples(df_processed)
    
    # Report summary statistics
    logger.info(f"Completed processing {processed_batches} batches")
    logger.info(f"Total entries marked for removal: {total_marked}")
    
    print(f"\nBatch Processing Summary:")
    print(f"  - Total batches in dataset: {len(all_batches)}")
    print(f"  - Batches with removals: {len(processed_set)}")
    print(f"  - Batches without removals: {len(no_removal_batches)}")
    print(f"  - Total entries marked as e-commerce: {total_marked}")
    
    return df_processed, total_marked


def visualize_emotion_distribution(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """
    Create visualization comparing emotion distribution before and after e-commerce removal.
    
    Purpose:
        Generates side-by-side bar charts showing how the emotion distribution
        changes after removing e-commerce reviews.
        
    Input:
        df_before: DataFrame before e-commerce removal
        df_after: DataFrame after e-commerce removal
    """
    
    logger.info("Creating emotion distribution comparison visualization")
    
    # Get emotion counts before and after cleaning
    emotions_before = df_before['emotion'].value_counts().sort_index()
    emotions_after = df_after['emotion'].value_counts().sort_index()
    
    # Create a DataFrame for side-by-side comparison
    comparison_data = pd.DataFrame({
        'Before': emotions_before,
        'After': emotions_after
    }).fillna(0)
    
    # Create visualization
    plt.figure(figsize=VIZ_SETTINGS['figsize_medium'])
    ax = comparison_data.plot(
        kind='bar', 
        figsize=VIZ_SETTINGS['figsize_medium'], 
        width=0.7,
        color=['#3498db', '#2ecc71'],
        edgecolor='black',
        linewidth=1
    )
    
    # Enhance visualization aesthetics
    plt.title('Emotion Distribution: Before vs After E-commerce Removal', 
              fontsize=VIZ_SETTINGS['title_fontsize'], 
              fontweight='bold', 
              pad=20)
    plt.xlabel('Emotion Category', fontsize=VIZ_SETTINGS['label_fontsize'], fontweight='bold')
    plt.ylabel('Number of Instances', fontsize=VIZ_SETTINGS['label_fontsize'], fontweight='bold')
    plt.xticks(rotation=30, fontsize=VIZ_SETTINGS['tick_fontsize'])
    plt.grid(axis='y', linestyle='--', alpha=VIZ_SETTINGS['grid_alpha'])
    
    # Add count labels on bars
    for i, emotion in enumerate(comparison_data.index):
        before_val = comparison_data.loc[emotion, 'Before']
        after_val = comparison_data.loc[emotion, 'After']
        
        # Add labels with counts
        plt.text(i-0.2, before_val + max(emotions_before)*0.02, 
                f'{int(before_val)}', ha='center', fontsize=10)
        plt.text(i+0.2, after_val + max(emotions_before)*0.02, 
                f'{int(after_val)}', ha='center', fontsize=10)
    
    # Add legend with enhanced styling
    legend = plt.legend(title='Dataset Version', loc='upper right', 
                       frameon=True, fontsize=12)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.8)
    
    plt.tight_layout()
    plt.show()

def visualize_final_distribution(df: pd.DataFrame) -> None:
    """
    Create professional visualization of the final emotion distribution.
    
    Purpose:
        Generates an enhanced visualization of the emotion distribution in
        the final cleaned dataset with detailed annotations and styling.
        
    Input:
        df: Final cleaned DataFrame after e-commerce removal
    """
    
    logger.info("Creating final emotion distribution visualization")
    
    # Create figure with professional styling
    plt.figure(figsize=VIZ_SETTINGS['figsize_large'])
    
    # Sort the data alphabetically by emotion for consistent visualization
    df_sorted = df.copy()
    emotion_order = sorted(df['emotion'].unique())
    
    # Set a professional color palette
    colors = sns.color_palette(VIZ_SETTINGS['palette'], len(emotion_order))
    
    # Create countplot with enhanced aesthetics and sorted categories
    # MODIFIED: Add hue parameter and set it equal to x parameter to fix the warning
    ax = sns.countplot(
        x='emotion',
        hue='emotion',  # Add hue parameter with same value as x
        data=df_sorted,
        order=emotion_order,  # Sort categories alphabetically
        hue_order=emotion_order,  # Keep the same order for hue
        palette=dict(zip(emotion_order, colors)),
        edgecolor='black',
        linewidth=1.2,
        legend=False  # Explicitly disable legend
    )
    
    # Customize appearance
    plt.title('Emotion Distribution in Final Cleaned Dataset', 
              fontsize=VIZ_SETTINGS['title_fontsize'], 
              fontweight='bold', 
              pad=20)
    plt.xlabel('Emotion Category', fontsize=VIZ_SETTINGS['label_fontsize'], fontweight='bold')
    plt.ylabel('Number of Instances', fontsize=VIZ_SETTINGS['label_fontsize'], fontweight='bold')
    plt.xticks(rotation=30, fontsize=VIZ_SETTINGS['tick_fontsize'], fontweight='bold')
    plt.yticks(fontsize=VIZ_SETTINGS['tick_fontsize'])
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=VIZ_SETTINGS['grid_alpha'])
    
    # Add value labels with percentages on top of each bar
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = round((height / total) * 100, 1)
        
        # Create annotated label with count and percentage
        ax.text(
            p.get_x() + p.get_width()/2.,
            height + 5,
            f'{int(height)}\n({percentage}%)',
            ha="center",
            fontsize=11,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.3", edgecolor='gray')
        )
    
    # Add a text box with dataset statistics
    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    textstr = (f'Total Dataset Size: {total}\n'
              f'Unique Emotions: {len(df["emotion"].unique())}\n'
              f'Most Common: {df["emotion"].value_counts().idxmax()} '
              f'({df["emotion"].value_counts().max()} instances)')
    plt.figtext(0.15, 0.02, textstr, fontsize=12, bbox=props)
    
    # Enhance border and adjust spacing
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.subplots_adjust(bottom=0.15, top=0.9)
    plt.tight_layout()
    plt.show()


def visualize_dataset_size_comparison(before_size: int, after_size: int) -> None:
    """
    Visualize the dataset size before and after e-commerce review removal.
    
    Purpose:
        Creates a bar chart comparing the dataset size before and after
        the cleaning process with detailed annotations.
        
    Input:
        before_size: Number of entries before cleaning
        after_size: Number of entries after cleaning
    """
    
    logger.info("Creating dataset size comparison visualization")
    
    # Calculate removal statistics
    removed = before_size - after_size
    removal_percentage = (removed / before_size) * 100 if before_size > 0 else 0
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sizes = [before_size, after_size]
    labels = ['Before Cleaning', 'After Cleaning']
    
    # Create bars with styling
    bars = plt.bar(
        labels, 
        sizes, 
        color=['#3498db', '#2ecc71'],
        edgecolor='black',
        linewidth=1,
        width=0.6
    )
    
    # Add styling and labels
    plt.title('Dataset Size Before and After E-commerce Review Removal', 
              fontsize=16, 
              fontweight='bold', 
              pad=20)
    plt.ylabel('Number of Entries', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add count labels on bars with boxes
    for i, (bar, count) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        plt.text(
            i, 
            height + 50, 
            f'{count}',
            ha='center',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.3", edgecolor='gray')
        )
    
    # Add arrow and text showing removal count
    plt.annotate(
        f'Removed: {removed} entries ({removal_percentage:.1f}%)', 
        xy=(0.5, (before_size + after_size)/2),
        xytext=(1.2, (before_size + after_size)/2), 
        arrowprops=dict(arrowstyle='->'),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
        fontsize=12
    )
    
    plt.tight_layout()
    plt.show()


def save_final_dataset(df: pd.DataFrame, file_path: str = OUTPUT_FILE) -> Tuple[pd.DataFrame, int]:
    """
    Remove all entries marked as e-commerce and save the final dataset.
    
    Purpose:
        Filters out all entries marked as e-commerce reviews, saves the clean
        dataset to a CSV file, and generates summary statistics.
        
    Input:
        df: DataFrame with marked e-commerce entries
        file_path: Path where the CSV file should be saved
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Final cleaned DataFrame
        - int: Number of removed entries
    """
    
    logger.info("Preparing final dataset for export")
    
    # Store initial count before removal
    initial_count = len(df)
    
    # Remove all entries marked as e-commerce
    final_df = df[~df['is_ecommerce']].copy()
    final_df = final_df.drop(columns=['is_ecommerce'])
    removed_count = initial_count - len(final_df)
    
    try:
        # Save cleaned dataset
        logger.info(f"Saving final dataset to {file_path}")
        final_df.to_csv(file_path, index=False, encoding=ENCODING)
        
        # Log save results
        logger.info(f"Dataset saved successfully with {len(final_df)} rows")
        
        # Summary statistics
        print("\n----------------------------------------")
        print("FINAL CLEANING SUMMARY")
        print("----------------------------------------")
        print(f"Starting dataset size: {initial_count} entries")
        print(f"E-commerce reviews removed: {removed_count} entries ({(removed_count/initial_count)*100:.1f}%)")
        print(f"Final dataset size: {len(final_df)} entries")
        print(f"Saved as: {file_path}")
        
        # Create download link in Colab
        print(f"\nDownloading cleaned dataset: {file_path}")
        files.download(file_path)
        
        return final_df, removed_count
        
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        print(f"Error: Failed to save dataset - {str(e)}")
        return final_df, removed_count


def main() -> None:
    """
    Main execution function for the e-commerce review removal pipeline.
    
    Purpose:
        Orchestrates the entire workflow from data loading through cleaning,
        visualization, and saving. Functions as the entry point for the program.
        
    Workflow:
        1. Load the previously cleaned dataset
        2. Initialize for e-commerce detection
        3. Process all batches using pre-identified data
        4. Save the final cleaned dataset
        5. Generate visualizations comparing before and after cleaning
    """
    
    logger.info("Starting Persian Emotion Dataset E-commerce Cleaning")
    
    # Step 1: Load the previously cleaned dataset
    df = load_dataset()
    if df is None:
        logger.error("Dataset loading failed. Exiting.")
        print("Error: Failed to load dataset. Please check the log for details.")
        return
    
    # Step 2: Initialize dataset for processing
    df, total_batches = initialize_dataset(df)
    
    # Step 3: Process all batches using pre-identified e-commerce entries
    df_processed, total_marked = process_all_batches(df, ECOMMERCE_BATCHES)
    
    # Step 4: Save the final cleaned dataset
    final_df, removed_count = save_final_dataset(df_processed)
    
    # Step 5: Generate visualizations
    print("\nGenerating visualizations for analysis...")
    visualize_dataset_size_comparison(len(df), len(final_df))
    visualize_emotion_distribution(df, final_df)
    visualize_final_distribution(final_df)
    
    # Final message
    logger.info("Persian Emotion Detection Dataset E-commerce Cleaning completed successfully")
    print("\nProcessing complete! The final cleaned dataset has been saved and downloaded.")
    print("Use emotion_ecommerce_cleaned.csv for your emotion analysis tasks.")


if __name__ == "__main__":
    main()
