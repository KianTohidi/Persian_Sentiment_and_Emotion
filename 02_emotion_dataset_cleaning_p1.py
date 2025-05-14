"""
Persian Emotion Detection Dataset Cleaning Part 1
-----------------------------------------

This script cleans and preprocesses the Armant Text Emotion dataset, which contains Persian social media and e-commerce texts for emotion detection.

Purpose:
    Process raw emotion-labeled Persian text data by removing duplicates, empty entries, 
    and outliers, while standardizing text and emotion labels for more reliable analysis.
    The script also generates visualization reports comparing data distributions before
    and after cleaning.

### IMPORTANT: Before running this code:
    1. Download the TSV dataset files from the GitHub website (link in the README)
    2. On the website, navigate to the "dataset" folder and locate the TSV files
    3. For each file, click on it and select "Download raw file" from the right menu
    4. During upload, press "Ctrl" while selecting to upload both files simultaneously

Input:
    Raw TSV files 'train.tsv' and 'test.tsv' with Persian social media and e-commerce text
    Expected format: Tab-separated values with text and emotion columns (no headers)

Output:
    - "emotion_cleaned.csv": Cleaned dataset with standardized format
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
        logging.FileHandler("emotion_cleaning.log")  # File output
    ]
)
logger = logging.getLogger(__name__)

# Constants for consistent reference
INPUT_FILES = ["train.tsv", "test.tsv"]
OUTPUT_FILE = "emotion_cleaned.csv"
ENCODING = "utf-8-sig"  # Standard encoding for Persian text

# Emotion label standardization mapping (sorted alphabetically)
EMOTION_MAPPING = {
    'angry': 'anger',
    'fear': 'fear',
    'happy': 'happiness',
    'hate': 'hate',
    'sad': 'sadness',
    'surprise': 'surprise',
    'other': 'other'
}

# Cleaning thresholds - centralized for easy modification
THRESHOLDS = {
    'min_text_length': 10,   # Minimum character length
    'max_word_count': 100    # Maximum number of words
}

def load_datasets(file_paths: List[str] = INPUT_FILES) -> Optional[List[pd.DataFrame]]:
    
    """
    Load multiple TSV datasets with proper encoding for Persian text.
    
    Purpose:
        Handles file upload in Colab environment and loads the datasets with the
        appropriate encoding for Persian text. Automatically handles renamed files.
        
    Input:
        file_paths: List of paths to the TSV files containing the datasets
        
    Returns:
        List[pd.DataFrame] or None: List of loaded datasets if successful, None otherwise
    """
    
    logger.info("Starting datasets load process")
    try:
        # Handle file upload in Colab
        logger.info(f"Prompting user to upload {', '.join(file_paths)}")
        print(f"Please upload the Armant Text Emotion dataset files ({', '.join(file_paths)}):")
        print("(Hold Ctrl key to select multiple files)")
        uploaded = files.upload()
        
        if not uploaded:
            logger.warning("No files were uploaded")
            return None
        
        # Load each dataset using flexible file matching
        dataframes = []
        uploaded_files = list(uploaded.keys())
        print(f"\nUploaded files: {', '.join(uploaded_files)}")
        
        for file_path in file_paths:
            # Find matching files (handle Colab's automatic renaming like "file.tsv" -> "file (1).tsv")
            file_base = file_path.split('.')[0]
            file_ext = file_path.split('.')[-1]
            
            # Find files that match the base name (with possible numbering) and extension
            matching_files = [f for f in uploaded_files if 
                             (file_base in f and f.endswith(file_ext))]
            
            if matching_files:
                matched_file = matching_files[0]  # Use the first match
                logger.info(f"Found matching file '{matched_file}' for expected file '{file_path}'")
                
                try:
                    logger.info(f"Loading dataset from {matched_file} with {ENCODING} encoding")
                    df = pd.read_csv(matched_file, 
                                    sep='\t', 
                                    encoding=ENCODING, 
                                    header=None, 
                                    names=['text', 'emotion'])
                    dataframes.append(df)
                    logger.info(f"Successfully loaded {matched_file}: {len(df)} rows")
                    print(f"Loaded {matched_file}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Error loading {matched_file}: {str(e)}")
                    print(f"Warning: Could not load {matched_file} - {str(e)}")
            else:
                logger.warning(f"No matching file found for {file_path}")
                print(f"Warning: No matching file found for {file_path}")
        
        if not dataframes:
            logger.error("No datasets could be loaded")
            return None
            
        return dataframes
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        print(f"Error: Failed to load datasets - {str(e)}")
        return None

def merge_datasets(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    
    """
    Merge multiple dataframes into a single dataframe for unified processing.
    
    Purpose:
        Combines multiple dataframes into one for consistent processing and
        provides basic statistics about the merge operation.
        
    Input:
        dataframes: List of DataFrames to merge
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    
    logger.info(f"Merging {len(dataframes)} datasets")
    
    # Track original sizes for reporting
    original_sizes = [len(df) for df in dataframes]
    total_original = sum(original_sizes)
    
    # Merge datasets
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Log merge results
    logger.info(f"Merged {len(dataframes)} datasets with a total of {total_original} rows")
    print(f"\nMerged datasets summary:")
    for i, size in enumerate(original_sizes):
        print(f"  - Dataset {i+1}: {size} rows ({(size/total_original)*100:.1f}%)")
    print(f"  - Total merged: {len(merged_df)} rows\n")
    
    return merged_df

def visualize_distribution(df: pd.DataFrame, category_column: str, title: str, 
                          figsize: Tuple[int, int] = (12, 6)) -> Dict[str, int]:
                              
    """
    Visualize the distribution of categories in the dataset.
    
    Purpose:
        Creates and displays a bar chart showing the distribution of categorical
        labels with count and percentage information.
        
    Input:
        df: DataFrame containing the category column
        category_column: Name of the column containing categories to visualize
        title: Title for the visualization
        figsize: Size of the figure (width, height) in inches
        
    Returns:
        Dict[str, int]: Category counts keyed by category label
    """
                              
    logger.info(f"Creating visualization: {title}")
    
    # Calculate category counts
    category_counts = df[category_column].value_counts()
    total = len(df)
    
    # Print statistics
    print(f"\n=== {title} ===")
    print(f"Total entries: {total}")
    
    # Sort emotions alphabetically for consistent reporting
    sorted_emotions = sorted(category_counts.keys())
    for category in sorted_emotions:
        count = category_counts[category]
        percentage = (count / total) * 100
        print(f"{category.capitalize()}: {count} entries ({percentage:.1f}%)")
    
    # Create visualization - sort categories alphabetically for consistent display
    plt.figure(figsize=figsize)
    sorted_order = sorted(category_counts.index)
    ax = sns.countplot(x=category_column, hue=category_column, data=df, 
                      palette='viridis', order=sorted_order, legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel(category_column.capitalize(), fontsize=14)
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
    
    return category_counts.to_dict()

def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    
    """
    Clean the dataset by removing duplicates, empty entries, and outliers.
    
    Purpose:
        Applies a series of cleaning operations to ensure data quality:
        - Removes entries with missing or empty text
        - Removes duplicate entries to prevent bias
        - Filters out too short or too long texts
        - Tracks statistics for each cleaning step
        
    Input:
        df: Raw DataFrame ready for cleaning
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Cleaned DataFrame
        - Dict[str, int]: Statistics about rows remaining after each cleaning step
    """
    
    logger.info("Starting dataset cleaning process")
    
    # Initialize statistics dictionary to track changes
    stats = {
        'original': len(df),
        'after_empty': 0,
        'after_duplicates': 0,
        'after_short': 0,
        'after_long': 0,
        'after_other': 0
    }
    
    # Make a copy to avoid modifying the input DataFrame
    df_clean = df.copy()
    
    # Step 1: Remove entries with missing or empty text
    logger.info("Removing entries with missing or empty text")
    df_clean = df_clean.dropna(subset=['text'])
    df_clean = df_clean[df_clean['text'].astype(str).str.strip() != '']
    stats['after_empty'] = len(df_clean)
    removed = stats['original'] - stats['after_empty']
    logger.info(f"Removed {removed} entries with missing or empty text")
    print(f"Removed empty texts: {removed}")
    print(f"Rows after removing empty texts: {stats['after_empty']}")
    
    # Step 2: Remove duplicate rows based on the text column
    logger.info("Removing duplicate entries")
    df_clean = df_clean.drop_duplicates(subset='text')
    stats['after_duplicates'] = len(df_clean)
    removed = stats['after_empty'] - stats['after_duplicates']
    logger.info(f"Removed {removed} duplicate entries")
    print(f"Removed duplicate texts: {removed}")
    print(f"Rows after removing duplicate texts: {stats['after_duplicates']}")
    
    # Step 3: Apply minimum text length threshold
    min_length = THRESHOLDS['min_text_length']
    logger.info(f"Removing texts shorter than {min_length} characters")
    df_clean = df_clean[df_clean['text'].astype(str).str.len() >= min_length]
    stats['after_short'] = len(df_clean)
    removed = stats['after_duplicates'] - stats['after_short']
    logger.info(f"Removed {removed} entries with texts shorter than {min_length} characters")
    print(f"Removed short texts: {removed}")
    print(f"Remaining rows after removing short texts: {stats['after_short']}")
    
    # Step 4: Remove entries with excessively long text
    max_words = THRESHOLDS['max_word_count']
    logger.info(f"Removing texts with more than {max_words} words")
    df_clean = df_clean[df_clean['text'].apply(lambda x: len(str(x).split()) <= max_words)]
    stats['after_long'] = len(df_clean)
    removed = stats['after_short'] - stats['after_long']
    logger.info(f"Removed {removed} entries with texts longer than {max_words} words")
    print(f"Removed long texts: {removed}")
    print(f"Remaining rows after removing long texts: {stats['after_long']}")
    
    print("\n")
    
    return df_clean, stats

def standardize_labels(df: pd.DataFrame, column: str, mapping: Dict[str, str]) -> pd.DataFrame:
    
    """
    Standardize categorical labels using a mapping dictionary.
    
    Purpose:
        Transforms categorical labels for consistency using a provided mapping,
        ensuring all labels follow a standard format.
        
    Input:
        df: DataFrame containing the column to standardize
        column: Name of the column containing labels to standardize
        mapping: Dictionary mapping original labels to standardized labels
        
    Returns:
        pd.DataFrame: DataFrame with standardized labels
    """
    
    logger.info(f"Standardizing labels in '{column}' column")
    
    # Make a copy to avoid modifying the input DataFrame
    df_standard = df.copy()
    
    # Ensure all labels are lowercase and stripped
    df_standard[column] = df_standard[column].astype(str).str.strip().str.lower()
    
    # Apply mapping to standardize labels
    df_standard[column] = df_standard[column].replace(mapping)
    
    # Log summary of standardization - display in alphabetical order
    label_counts = df_standard[column].value_counts()
    sorted_labels = sorted(label_counts.index)
    sorted_labels_str = ', '.join(sorted_labels)
    logger.info(f"Standardized labels: {sorted_labels_str}")
    
    return df_standard

def remove_category(df: pd.DataFrame, column: str, category: str) -> Tuple[pd.DataFrame, int]:
    
    """
    Remove entries with a specific category label.
    
    Purpose:
        Filters out entries with a specified category label and reports on
        the number of entries removed.
        
    Input:
        df: DataFrame containing the category column
        column: Name of the column containing categories
        category: The category value to remove
        
    Returns:
        Tuple containing:
        - pd.DataFrame: DataFrame with category entries removed
        - int: Number of entries removed
    """
    
    logger.info(f"Removing entries with '{category}' in '{column}' column")
    
    # Count entries before removal
    before_count = len(df)
    
    # Remove entries with the specified category
    df_filtered = df[df[column] != category]
    
    # Calculate number of entries removed
    removed = before_count - len(df_filtered)
    
    # Log results
    logger.info(f"Removed {removed} entries with '{category}' in '{column}' column")
    print(f"Removed '{category}' labels: {removed}")
    print(f"Remaining rows after removing '{category}' category: {len(df_filtered)}")
    print("\n")
    
    return df_filtered, removed

def visualize_cleaning_steps(steps: List[str], counts: List[int]) -> None:
    
    """
    Visualize the impact of each cleaning step on dataset size.
    
    Purpose:
        Creates a bar chart showing how the dataset size changes after each
        cleaning operation to visualize the impact of the cleaning process.
        
    Input:
        steps: List of cleaning step names
        counts: List of dataset sizes after each step
    """
    
    logger.info("Creating cleaning steps visualization")
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(steps, counts, color='teal')
    plt.title('Dataset Size After Each Cleaning Step', fontsize=16)
    plt.xlabel('Cleaning Step', fontsize=14)
    plt.ylabel('Number of Entries', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                f'{count}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    print("\n")

def save_dataset(df: pd.DataFrame, file_path: str = OUTPUT_FILE) -> bool:
    
    """
    Save the cleaned dataset to a CSV file with appropriate encoding.
    
    Purpose:
        Exports the cleaned dataset to a CSV file and creates a download link
        in Colab environment for user access.
        
    Input:
        df: DataFrame to save
        file_path: Path where the CSV file should be saved
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    
    try:
        logger.info(f"Saving cleaned dataset to {file_path}")
        df.to_csv(file_path, index=False, encoding=ENCODING)
        logger.info(f"Dataset saved successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Create download link in Colab
        print(f"\nDownloading cleaned dataset: {file_path}")
        files.download(file_path)
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        print(f"Error: Failed to save dataset - {str(e)}")
        return False

def print_cleaning_summary(stats: Dict[str, int]) -> None:
    
    """
    Print a summary of the cleaning process.
    
    Purpose:
        Formats and displays a comprehensive summary of the data cleaning
        process, including counts and percentages of removed entries at each step.
        
    Input:
        stats: Dictionary containing statistics about the cleaning process
    """
    
    print("\n=== CLEANING SUMMARY ===")
    print(f"Starting dataset size: {stats['original']} entries")
    
    steps = [
        ('Empty texts removed', stats['original'] - stats['after_empty']),
        ('Duplicates removed', stats['after_empty'] - stats['after_duplicates']),
        ('Short texts removed', stats['after_duplicates'] - stats['after_short']),
        ('Long texts removed', stats['after_short'] - stats['after_long']),
        ('Other labels removed', stats['after_long'] - stats['after_other'])
    ]
    
    print(f"Entries removed:")
    for step_name, count in steps:
        percentage = (count / stats['original']) * 100 if stats['original'] > 0 else 0
        print(f"  - {step_name}: {count} entries ({percentage:.1f}%)")
    
    total_removed = stats['original'] - stats['after_other']
    removal_percentage = (total_removed / stats['original']) * 100 if stats['original'] > 0 else 0
    
    print(f"Final dataset size: {stats['after_other']} entries")
    print(f"Total removed: {total_removed} entries ({removal_percentage:.1f}%)")
    print("\n")

def main() -> None:
    
    """
    Main execution function for the Persian emotion dataset cleaning pipeline.
    
    Purpose:
        Orchestrates the entire workflow from data loading through cleaning,
        visualization, and saving. Functions as the entry point for the program.
        
    Workflow:
        1. Load datasets
        2. Merge datasets
        3. Visualize initial distribution
        4. Clean dataset
        5. Standardize emotion labels
        6. Remove 'other' category
        7. Visualize cleaned distribution
        8. Visualize cleaning steps impact
        9. Save cleaned dataset
    """
    
    logger.info("Starting Persian Emotion Detection Dataset Cleaning")
    
    # Step 1: Load the datasets
    datasets = load_datasets()
    if not datasets:
        logger.error("Dataset loading failed. Exiting.")
        return
    
    # Step 2: Merge datasets for unified processing
    merged_df = merge_datasets(datasets)
    
    # Step 3: Visualize emotion distribution before cleaning
    emotions_before = visualize_distribution(
        merged_df, 
        'emotion', 
        "Emotion Distribution - Before Cleaning"
    )
    
    # Step 4: Clean the dataset
    merged_df, cleaning_stats = clean_dataset(merged_df)
    
    # Step 5: Standardize emotion labels
    merged_df = standardize_labels(merged_df, 'emotion', EMOTION_MAPPING)
    
    # Step 6: Remove 'other' emotion category
    merged_df, other_removed = remove_category(merged_df, 'emotion', 'other')
    cleaning_stats['after_other'] = len(merged_df)
    
    # Step 7: Visualize emotion distribution after cleaning
    emotions_after = visualize_distribution(
        merged_df, 
        'emotion', 
        "Emotion Distribution - After Cleaning"
    )
    
    # Step 8: Print cleaning summary
    print_cleaning_summary(cleaning_stats)
    
    # Step 9: Visualize the impact of cleaning on dataset size
    cleaning_steps = ['Original', 'After Empty Removal', 'After Duplicates Removal', 
                     'After Short Text Removal', 'After Long Text Removal', 'After Other Label Removal']
    counts = [
        cleaning_stats['original'],
        cleaning_stats['after_empty'],
        cleaning_stats['after_duplicates'],
        cleaning_stats['after_short'],
        cleaning_stats['after_long'],
        cleaning_stats['after_other']
    ]
    
    visualize_cleaning_steps(cleaning_steps, counts)
    
    # Step 10: Save the cleaned dataset
    save_success = save_dataset(merged_df)
    
    # Final message
    if save_success:
        logger.info("Persian Emotion Detection Dataset Cleaning completed successfully")
        print("\nProcessing complete! The cleaned dataset has been saved and downloaded.")
    else:
        logger.warning("Persian Emotion Detection Dataset Cleaning completed with warnings")
        print("\nProcessing complete with warnings. Please check the log for details.")

if __name__ == "__main__":
    main()
