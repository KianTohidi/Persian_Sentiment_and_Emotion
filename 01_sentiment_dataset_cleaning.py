"""
Persian Sentiment Analysis Dataset Cleaning
-------------------------------------------

This script cleans and preprocesses the Persian Sentiment Analysis dataset, which contains Persian Instagram comments for sentiment analysis.

Purpose:
    Processes raw sentiment-labeled data by removing duplicates, empty entries, and outliers,
    while standardizing text and sentiment labels for more reliable analysis. The script also
    generates visualization reports to compare data distributions before and after cleaning.

### IMPORTANT: Before running this code:
  1. Download the CSV dataset from the Kaggle website (link in the README)
   2. Extract the zip file
    3. Rename the file from "Instagram labeled comments.csv" to "sentiment.csv"

Input:
    Raw CSV file "sentiment.csv" with Persian Instagram comments and numerical sentiment labels
    Expected format: CSV with 'comment' and 'sentiment' columns (sentiment values: 1, 0, -1)


Output:
    - "sentiment_cleaned.csv": Cleaned dataset with standardized format
    - Visualization reports showing sentiment distribution before and after cleaning
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
        logging.FileHandler("sentiment_cleaning.log")  # File output
    ]
)
logger = logging.getLogger(__name__)

# Constants for consistent reference
INPUT_FILE = "sentiment.csv"
OUTPUT_FILE = "sentiment_cleaned.csv"
ENCODING = "utf-8-sig"  # Standard encoding for Persian text
SENTIMENT_MAPPING = {
    1: 'positive',
    0: 'neutral',
    -1: 'negative'
}

# Cleaning thresholds - centralized for easy modification
THRESHOLDS = {
    'min_text_length': 10,   # Minimum character length
    'max_word_count': 100    # Maximum number of words
}

def load_dataset(file_path: str = INPUT_FILE) -> Optional[pd.DataFrame]:
    
    """
    Load the dataset from a CSV file with proper encoding for Persian text.
    
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
        print(f"Please upload the Persian Sentiment Analysis dataset file ({file_path}):")
        uploaded = files.upload()
        
        if not uploaded:
            logger.warning("No file was uploaded")
            return None
            
        # Check if the expected file is in the uploaded files
        if file_path not in uploaded:
            # If the file isn't found by the expected name, use the first uploaded file
            file_path = list(uploaded.keys())[0]
            logger.info(f"Using uploaded file: {file_path}")
        
        # Attempt to load the dataset with appropriate encoding
        logger.info(f"Loading dataset from {file_path} with {ENCODING} encoding")
        df = pd.read_csv(file_path, encoding=ENCODING)
        
        # Log basic information about the loaded dataset
        logger.info(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"\nDataset loaded successfully with {len(df)} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        print(f"Error: Failed to load dataset - {str(e)}")
        return None

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Perform initial preprocessing on the dataset to standardize column names and values.
    
    Purpose:
        Standardizes the dataset structure by removing unnecessary columns,
        renaming columns for consistency, and converting numerical sentiment
        labels to text format.
        
    Input:
        df: Raw DataFrame with original column structure
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with standardized structure
    """
    
    logger.info("Starting dataset preprocessing")
    
    # Store original row count for reference
    original_count = len(df)
    logger.info(f"Original dataset size: {original_count} rows")
    
    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Step 1: Drop unnamed index columns that might have been added during export
    unnamed_cols = [col for col in df_processed.columns if 'Unnamed' in col]
    if unnamed_cols:
        logger.info(f"Removing {len(unnamed_cols)} unnamed column(s): {unnamed_cols}")
        df_processed = df_processed.drop(columns=unnamed_cols)
    
    # Step 2: Rename columns for consistency with standard NLP conventions
    if 'comment' in df_processed.columns:
        logger.info("Renaming 'comment' column to 'text' for consistency")
        df_processed = df_processed.rename(columns={'comment': 'text'})
    
    # Step 3: Convert numerical sentiment labels to text format
    if 'sentiment' in df_processed.columns:
        logger.info("Converting numerical sentiment labels to text format")
        df_processed['sentiment'] = df_processed['sentiment'].map(SENTIMENT_MAPPING)
        
        # Check for any unmapped values
        unmapped = df_processed[~df_processed['sentiment'].isin(SENTIMENT_MAPPING.values())]
        if not unmapped.empty:
            logger.warning(f"Found {len(unmapped)} rows with unmapped sentiment values")
            
    logger.info("Preprocessing completed")
    return df_processed

def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    
    """
    Clean the dataset by removing duplicates, empty entries, and outliers.
    
    Purpose:
        Applies a series of cleaning operations to ensure data quality:
        - Removes duplicate entries to prevent bias
        - Removes entries with missing or empty text
        - Filters out too short or too long texts
        - Tracks statistics for each cleaning step
        
    Input:
        df: Preprocessed DataFrame ready for cleaning
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Cleaned DataFrame
        - Dict[str, int]: Statistics about rows removed in each cleaning step
    """
    logger.info("Starting dataset cleaning process")
    
    # Initialize statistics dictionary to track changes
    stats = {
        'original': len(df),
        'duplicates_removed': 0,
        'empty_removed': 0,
        'short_removed': 0,
        'long_removed': 0,
        'final': 0
    }
    
    # Make a copy to avoid modifying the input DataFrame
    df_clean = df.copy()
    
    # Step 1: Remove duplicate rows based on the text column
    logger.info("Removing duplicate entries")
    before_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset='text')
    stats['duplicates_removed'] = before_count - len(df_clean)
    logger.info(f"Removed {stats['duplicates_removed']} duplicate entries")
    
    # Step 2: Remove entries with missing or empty text
    logger.info("Removing entries with missing or empty text")
    before_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['text'])
    df_clean = df_clean[df_clean['text'].astype(str).str.strip() != '']
    stats['empty_removed'] = before_count - len(df_clean)
    logger.info(f"Removed {stats['empty_removed']} entries with missing or empty text")
    
    # Step 3: Apply minimum text length threshold
    logger.info(f"Removing texts shorter than {THRESHOLDS['min_text_length']} characters")
    before_count = len(df_clean)
    df_clean = df_clean[df_clean['text'].astype(str).str.len() >= THRESHOLDS['min_text_length']]
    stats['short_removed'] = before_count - len(df_clean)
    logger.info(f"Removed {stats['short_removed']} entries with texts shorter than {THRESHOLDS['min_text_length']} characters")
    
    # Step 4: Remove entries with excessively long text
    logger.info(f"Removing texts with more than {THRESHOLDS['max_word_count']} words")
    before_count = len(df_clean)
    df_clean = df_clean[df_clean['text'].apply(lambda x: len(str(x).split()) <= THRESHOLDS['max_word_count'])]
    stats['long_removed'] = before_count - len(df_clean)
    logger.info(f"Removed {stats['long_removed']} entries with texts longer than {THRESHOLDS['max_word_count']} words")
    
    # Update final count
    stats['final'] = len(df_clean)
    total_removed = stats['original'] - stats['final']
    removal_percentage = (total_removed / stats['original']) * 100 if stats['original'] > 0 else 0
    
    # Log summary of cleaning process
    logger.info(f"Cleaning completed: {total_removed} entries removed ({removal_percentage:.1f}%)")
    logger.info(f"Final dataset size: {stats['final']} entries")
    
    return df_clean, stats

def visualize_sentiment_distribution(df: pd.DataFrame, title: str, figsize: Tuple[int, int] = (10, 6)) -> Dict[str, int]:
    
    """
    Visualize the sentiment distribution in the dataset.
    
    Purpose:
        Creates and displays a bar chart showing the distribution of sentiment
        labels with count and percentage information.
        
    Input:
        df: DataFrame containing 'sentiment' column
        title: Title for the visualization
        figsize: Size of the figure (width, height) in inches
        
    Returns:
        Dict[str, int]: Sentiment counts keyed by sentiment label
    """
    
    logger.info(f"Creating visualization: {title}")
    
    # Calculate sentiment counts
    sentiment_counts = df['sentiment'].value_counts()
    total = len(df)
    
    # Create visualization
    plt.figure(figsize=figsize)
    ax = sns.countplot(x='sentiment', hue='sentiment', data=df, palette='viridis', legend=False)
    plt.title(title, fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=12)
    
    # Print sentiment distribution statistics
    print(f"\n=== {title} ===")
    print(f"Total entries: {total}")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        print(f"{sentiment.capitalize()}: {count} entries ({percentage:.1f}%)")
    
    plt.tight_layout()
    plt.show()
    print("\n")
    
    return sentiment_counts.to_dict()

def compare_distributions(before_counts: Dict[str, int], after_counts: Dict[str, int], 
                         before_total: int, after_total: int, figsize: Tuple[int, int] = (12, 7)) -> None:
                             
    """
    Compare sentiment distributions before and after cleaning.
    
    Purpose:
        Creates a side-by-side bar chart comparing sentiment distributions
        before and after cleaning, with percentage calculations.
        
    Input:
        before_counts: Dictionary of sentiment counts before cleaning
        after_counts: Dictionary of sentiment counts after cleaning
        before_total: Total number of entries before cleaning
        after_total: Total number of entries after cleaning
        figsize: Size of the figure (width, height) in inches
    """
                             
    logger.info("Creating comparison visualization for before and after cleaning")
    
    # Calculate total reduction
    total_reduction = before_total - after_total
    reduction_percentage = (total_reduction / before_total) * 100 if before_total > 0 else 0
    
    print("\n=== DISTRIBUTION CHANGE ANALYSIS ===")
    print(f"Total reduction: {total_reduction} entries ({reduction_percentage:.1f}%)")
    
    # Get list of all sentiment labels
    sentiments = list(SENTIMENT_MAPPING.values())
    
    # Calculate percentages
    before_percentages = [(before_counts.get(s, 0) / before_total) * 100 for s in sentiments]
    after_percentages = [(after_counts.get(s, 0) / after_total) * 100 for s in sentiments]
    
    # Set up the bar chart positions
    x = np.arange(len(sentiments))
    width = 0.35
    
    # Create the bars
    fig, ax = plt.subplots(figsize=figsize)
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
    
    # Calculate and display percentage point changes
    print("\nPercentage point changes:")
    for i, sentiment in enumerate(sentiments):
        change = after_percentages[i] - before_percentages[i]
        print(f"{sentiment.capitalize()}: {change:+.1f} percentage points")
        
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
        process, including counts and percentages of removed entries.
        
    Input:
        stats: Dictionary containing statistics about the cleaning process
    """
    
    print("\n=== CLEANING SUMMARY ===")
    print(f"Starting dataset size: {stats['original']} entries")
    print(f"Entries removed:")
    print(f"  - Duplicates: {stats['duplicates_removed']} entries")
    print(f"  - Empty texts: {stats['empty_removed']} entries")
    print(f"  - Short texts: {stats['short_removed']} entries")
    print(f"  - Long texts: {stats['long_removed']} entries")
    
    total_removed = stats['original'] - stats['final']
    removal_percentage = (total_removed / stats['original']) * 100 if stats['original'] > 0 else 0
    
    print(f"Final dataset size: {stats['final']} entries")
    print(f"Total removed: {total_removed} entries ({removal_percentage:.1f}%)")
    print("\n")

def main() -> None:
    
    """
    Main execution function for the Persian sentiment dataset cleaning pipeline.
    
    Purpose:
        Orchestrates the entire workflow from data loading through cleaning,
        visualization, and saving. Functions as the entry point for the program.
        
    Workflow:
        1. Load dataset (via load_dataset)
        2. Preprocess for standardization (via preprocess_dataset)
        3. Visualize initial distribution
        4. Clean dataset (via clean_dataset)
        5. Visualize cleaned distribution
        6. Compare distributions
        7. Save cleaned dataset
    """
    
    logger.info("Starting Persian Sentiment Analysis Dataset Cleaning")
    
    # Step 1: Load the dataset
    df = load_dataset()
    if df is None:
        logger.error("Dataset loading failed. Exiting.")
        return
    
    # Step 2: Preprocess the dataset
    df_preprocessed = preprocess_dataset(df)
    
    # Step 3: Visualize sentiment distribution before cleaning
    before_counts = visualize_sentiment_distribution(
        df_preprocessed, 
        "Sentiment Distribution - Before Cleaning"
    )
    before_total = len(df_preprocessed)
    
    # Step 4: Clean the dataset
    df_cleaned, cleaning_stats = clean_dataset(df_preprocessed)
    
    # Step 5: Print cleaning summary
    print_cleaning_summary(cleaning_stats)
    
    # Step 6: Visualize sentiment distribution after cleaning
    after_counts = visualize_sentiment_distribution(
        df_cleaned, 
        "Sentiment Distribution - After Cleaning"
    )
    after_total = len(df_cleaned)
    
    # Step 7: Compare distributions before and after cleaning
    compare_distributions(
        before_counts, 
        after_counts, 
        before_total, 
        after_total
    )
    
    # Step 8: Save the cleaned dataset
    save_success = save_dataset(df_cleaned)
    
    # Final message
    if save_success:
        logger.info("Persian Sentiment Analysis Dataset Cleaning completed successfully")
        print("\nProcessing complete! The cleaned dataset has been saved and downloaded.")
    else:
        logger.warning("Persian Sentiment Analysis Dataset Cleaning completed with warnings")
        print("\nProcessing complete with warnings. Please check the log for details.")

if __name__ == "__main__":
    main()
