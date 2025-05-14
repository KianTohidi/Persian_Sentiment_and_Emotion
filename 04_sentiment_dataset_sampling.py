"""
Persian Sentiment Analysis - Reproducible Balanced Dataset Creation
------------------------------------------------------------------

This script creates a balanced dataset for Persian sentiment analysis by sampling an equal 
number of entries from each sentiment class with GUARANTEED reproducibility.

Purpose:
    Creates a balanced dataset from an imbalanced sentiment-labeled dataset by sampling equal
    numbers from each sentiment class (positive, neutral, negative), with strict controls
    to ensure exact reproducibility across different runs and environments.

### IMPORTANT: Before running this code:
    1. Make sure you have already run the dataset cleaning script
    2. Have the cleaned CSV file ready for upload (sentiment_cleaned.csv)

Input:
    Cleaned CSV file with Persian texts and sentiment labels (sentiment_cleaned.csv)
    Expected format: CSV with 'text' and 'sentiment' columns (sentiment values: 'positive', 'neutral', 'negative')

Output:
    - "sentiment_balanced.csv": Reproducible balanced dataset with equal representation of each sentiment class
    - Statistical reports comparing original and balanced distributions
    - Percentage change analysis between original and balanced datasets

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - matplotlib (tested with version 3.10.0)
    - google.colab (for file upload/download in Colab environment)
    
Expected Checksum Values:
    After first running this code successfully, update these values in the constants section:
    - Dataset creation checksum: Will be a consistent MD5 hash string
    - Final verification checksum: Will be a consistent MD5 hash string
"""

"""
Reproducibility Guarantee and Limitations
-----------------------------------------
This code is designed to guarantee identical output every time, regardless of environment or user, as long as:
- The input file is the same
- The sample size parameter is the same
- The defined sentiment classes remain the same

Potential Limitations to Perfect Reproducibility:
1. Library Version Differences: Using different versions of pandas/numpy may affect results if these 
   libraries fundamentally change their random number implementation between versions.
   (Current implementation tested with pandas 2.2.2 and numpy 2.0.2)

2. Hardware Variations: In extremely rare cases, hardware-specific floating point differences might 
   affect operations that depend on floating point arithmetic, potentially leading to slightly 
   different sorting or sampling results.

3. Code Modifications: Any modification to the sampling logic, sorting operations, or random seed 
   management will likely affect reproducibility.

To verify reproducibility, this code generates a checksum of the final dataset that should be
identical across different runs with the same input parameters.
"""

import os
import random
import logging
import pandas as pd
import numpy as np
import io
import hashlib  # Added for deterministic checksums
from typing import Dict, Tuple, Optional, Union, List
from google.colab import files
import matplotlib.pyplot as plt

# Configure logging for detailed process tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("sentiment_balancing.log")  # File output
    ]
)
logger = logging.getLogger(__name__)

# Constants for consistent reference
INPUT_FILE = "sentiment_cleaned.csv"
OUTPUT_FILE = "sentiment_balanced.csv"
ENCODING = "utf-8-sig"  # Standard encoding for Persian text
RANDOM_SEED = 42  # Fixed seed for reproducibility
SAMPLE_SIZE = 300  # Default number of entries to sample from each sentiment class
SENTIMENT_CLASSES = ['positive', 'neutral', 'negative']  # Expected sentiment classes in a consistent order

# Checksum validation constants 
# These will be MD5 hash strings rather than integer values
EXPECTED_CREATION_CHECKSUM = "3ae7c9d128c73ee3a7eba539e1e2541a"
EXPECTED_VERIFICATION_CHECKSUM = "804f187fefc3db0f0258a8637d1ecb94"

def set_all_seeds(seed_value):
    
    """
    Set all seeds to ensure reproducibility across different environments.
    
    Args:
        seed_value: Seed value to use for all random number generators
    """
    
    # Set seeds for Python's random module
    random.seed(seed_value)
    
    # Set seeds for NumPy
    np.random.seed(seed_value)
    
    # Set seeds for Python's hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    logger.info(f"All random seeds set to {seed_value} for reproducibility")

def load_dataset() -> Optional[pd.DataFrame]:
    
    """
    Load the cleaned sentiment dataset from a CSV file with proper encoding for Persian text.
    
    Purpose:
        Handles file upload in Colab environment and loads the dataset with the
        appropriate encoding for Persian text.
        
    Returns:
        pd.DataFrame or None: Loaded dataset if successful, None otherwise
    """
    
    logger.info("Starting dataset load process")
    try:
        # Handle file upload in Colab
        logger.info(f"Prompting user to upload cleaned sentiment dataset")
        print(f"Please upload your sentiment cleaned CSV file: ({INPUT_FILE})")
        uploaded = files.upload()
        
        if not uploaded:
            logger.warning("No file was uploaded")
            return None
            
        # Get the filename of the uploaded file
        file_name = list(uploaded.keys())[0]
        logger.info(f"File '{file_name}' uploaded successfully")
        
        # Read the CSV file with appropriate encoding for Persian text
        logger.info(f"Loading dataset from {file_name} with {ENCODING} encoding")
        df = pd.read_csv(io.BytesIO(uploaded[file_name]), encoding=ENCODING)
        
        # Log basic information about the loaded dataset
        logger.info(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"\nDataset loaded successfully with {len(df)} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        print(f"Error: Failed to load dataset - {str(e)}")
        return None

def analyze_sentiment_distribution(df: pd.DataFrame, title: str = "Dataset") -> Dict[str, int]:
    
    """
    Analyze and display the sentiment distribution in the dataset.
    
    Purpose:
        Calculates and displays the distribution of sentiment labels in the dataset,
        including counts and percentages for each sentiment class.
        
    Args:
        df: DataFrame containing a 'sentiment' column
        title: Description of the dataset for display purposes
        
    Returns:
        Dict[str, int]: Dictionary of sentiment counts keyed by sentiment label
    """
    
    logger.info(f"Analyzing sentiment distribution for: {title}")
    
    # Calculate sentiment counts
    sentiment_counts = df['sentiment'].value_counts()
    total = len(df)
    
    # Display information
    print(f"\n=== {title.upper()} INFORMATION ===")
    print(f"Total number of entries: {total}")
    
    if title == "Original Dataset":
        print(f"Columns: {', '.join(df.columns)}")
    
    # Show the distribution of sentiment labels
    print("\nSentiment Distribution:")
    # Use a consistent order for sentiment classes to ensure reproducible output
    counts_dict = sentiment_counts.to_dict()
    for sentiment in SENTIMENT_CLASSES:
        count = counts_dict.get(sentiment, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{sentiment.capitalize()}: {count} entries ({percentage:.1f}%)")
    
    return counts_dict

def create_reproducible_balanced_dataset(df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    
    """
    Create a balanced dataset with guaranteed reproducibility.
    
    Purpose:
        Ensures equal representation of each sentiment class by sampling a specified
        number of entries from each class, with strict controls for reproducibility.
        
    Args:
        df: DataFrame containing text and sentiment columns
        sample_size: Number of entries to sample from each sentiment class
        
    Returns:
        pd.DataFrame: Balanced dataset with equal representation of each sentiment class
        
    Expected Checksum:
        Will generate a deterministic MD5 hash that should be consistent across all
        environments when using the same input data and parameters.
    """
    
    logger.info(f"Creating balanced dataset with {sample_size} samples per sentiment class")
    
    # Create a fresh DataFrame to hold our samples
    sampled_data = pd.DataFrame(columns=df.columns)
    
    # Reset the random seed before starting any sampling
    # This ensures the sampling process always starts from the same state
    set_all_seeds(RANDOM_SEED)
    
    # Process each sentiment category in a consistent order
    for sentiment in SENTIMENT_CLASSES:
        # Filter the dataframe for the current sentiment
        sentiment_df = df[df['sentiment'] == sentiment].copy()
        
        # Sort the filtered data by a consistent criterion to ensure deterministic ordering
        # This step is critical for reproducibility - we first set a deterministic order
        # before applying random sampling
        sentiment_df = sentiment_df.sort_values(by=['text']).reset_index(drop=True)
        
        # Check if we have enough data for this sentiment
        if len(sentiment_df) < sample_size:
            logger.warning(f"Not enough data for '{sentiment}'. Only {len(sentiment_df)} entries available.")
            print(f"\nWarning: Not enough data for '{sentiment}'. Only {len(sentiment_df)} entries available.")
            # Take all available data for this sentiment
            sampled_sentiment = sentiment_df
        else:
            # Reset the random seed immediately before sampling
            # This step ensures each sentiment class sampling starts from the same seed state
            np.random.seed(RANDOM_SEED)
            
            # Randomly sample the specified number of entries
            logger.info(f"Sampling {sample_size} entries for '{sentiment}' sentiment")
            sampled_sentiment = sentiment_df.sample(n=sample_size, random_state=RANDOM_SEED)
        
        # Add to the sampled dataframe
        sampled_data = pd.concat([sampled_data, sampled_sentiment])
    
    # For the final shuffle, reset the random seed once more to ensure reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Sort by text first to establish a deterministic order before shuffling
    sampled_data = sampled_data.sort_values(by=['text']).reset_index(drop=True)
    
    # Now shuffle with the reset random seed for reproducible randomization
    logger.info("Shuffling the balanced dataset in a reproducible manner")
    sampled_data = sampled_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Add a deterministic checksum to verify dataset reproducibility
    # Using MD5 hash instead of Python's built-in hash() for consistency across environments
    dataset_str = ''.join(sampled_data['text'].astype(str).tolist())
    checksum = hashlib.md5(dataset_str.encode('utf-8')).hexdigest()
    
    logger.info(f"Balanced dataset created with checksum: {checksum}")
    print(f"\nCreated balanced dataset with reproducibility checksum: {checksum}")
    
    # Validate checksum if expected value is provided
    if EXPECTED_CREATION_CHECKSUM is not None:
        if checksum == EXPECTED_CREATION_CHECKSUM:
            logger.info("Creation checksum validation: PASSED ✓")
            print("Creation checksum validation: PASSED ✓")
        else:
            logger.warning(f"Creation checksum validation: FAILED ✗ (Expected: {EXPECTED_CREATION_CHECKSUM}, Got: {checksum})")
            print(f"Creation checksum validation: FAILED ✗ (Expected: {EXPECTED_CREATION_CHECKSUM}, Got: {checksum})")
    
    logger.info(f"Balanced dataset created with {len(sampled_data)} total entries")
    
    return sampled_data

def visualize_comparison(original_counts: Dict[str, int], balanced_counts: Dict[str, int], 
                         original_total: int, balanced_total: int) -> None:
                             
    """
    Visualize the comparison between original and balanced datasets.
    
    Purpose:
        Creates a bar chart comparing the sentiment distributions in the
        original and balanced datasets as percentages.
        
    Args:
        original_counts: Dictionary of sentiment counts in the original dataset
        balanced_counts: Dictionary of sentiment counts in the balanced dataset
        original_total: Total number of entries in the original dataset
        balanced_total: Total number of entries in the balanced dataset
    """
                             
    logger.info("Creating comparison visualization for original and balanced datasets")
    
    # Reset the random seed before plotting for consistent colors/styles
    np.random.seed(RANDOM_SEED)
    
    # Get list of all sentiment labels (ensuring consistent order)
    sentiments = SENTIMENT_CLASSES
    
    # Calculate percentages
    original_percentages = [(original_counts.get(s, 0) / original_total) * 100 if original_total > 0 else 0 for s in sentiments]
    balanced_percentages = [(balanced_counts.get(s, 0) / balanced_total) * 100 if balanced_total > 0 else 0 for s in sentiments]
    
    # Set up the bar chart positions
    x = np.arange(len(sentiments))
    width = 0.35
    
    # Create the bars
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, original_percentages, width, label='Original Dataset (%)', color='skyblue')
    rects2 = ax.bar(x + width/2, balanced_percentages, width, label='Balanced Dataset (%)', color='lightgreen')
    
    # Add labels and title
    ax.set_xlabel('Sentiment', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.set_title('Sentiment Distribution Comparison: Original vs Balanced (%)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sentiments])
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
    
    plt.tight_layout()
    plt.show()

def save_balanced_dataset(df: pd.DataFrame, file_path: str = OUTPUT_FILE) -> bool:
    
    """
    Save the balanced dataset to a CSV file with proper encoding.
    
    Purpose:
        Exports the balanced dataset to a CSV file and creates a download link
        in Colab environment for user access.
        
    Args:
        df: DataFrame to save
        file_path: Path where the CSV file should be saved
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    
    try:
        logger.info(f"Saving balanced dataset to {file_path}")
        df.to_csv(file_path, index=False, encoding=ENCODING)
        logger.info(f"Dataset saved successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Create download link in Colab
        logger.info("Creating download link for balanced dataset")
        files.download(file_path)
        print(f"\nFile '{file_path}' has been created and downloaded successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        print(f"Error: Failed to save balanced dataset - {str(e)}")
        return False

def print_comparison_summary(original_counts: Dict[str, int], balanced_counts: Dict[str, int],
                           original_total: int, balanced_total: int) -> None:
                               
    """
    Print a summary comparing the original and balanced datasets.
    
    Purpose:
        Formats and displays a comprehensive comparison between the original
        and balanced datasets, including size reduction and distribution changes.
        
    Args:
        original_counts: Dictionary of sentiment counts in the original dataset
        balanced_counts: Dictionary of sentiment counts in the balanced dataset
        original_total: Total number of entries in the original dataset
        balanced_total: Total number of entries in the balanced dataset
    """
                               
    logger.info("Generating comparison summary between original and balanced datasets")
    
    # Calculate overall reduction
    reduction = original_total - balanced_total
    reduction_percentage = (reduction / original_total) * 100 if original_total > 0 else 0
    
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Original dataset size: {original_total} entries")
    print(f"Balanced dataset size: {balanced_total} entries")
    print(f"Reduction: {reduction} entries ({reduction_percentage:.1f}%)")
    
    # Calculate and display the change in distribution
    print("\nPercentage point changes:")
    for sentiment in SENTIMENT_CLASSES:
        original_percentage = (original_counts.get(sentiment, 0) / original_total) * 100 if original_total > 0 else 0
        balanced_percentage = (balanced_counts.get(sentiment, 0) / balanced_total) * 100 if balanced_total > 0 else 0
        change = balanced_percentage - original_percentage
        print(f"{sentiment.capitalize()}: {change:+.1f} percentage points")

def verify_reproducibility(df: pd.DataFrame) -> str:
    
    """
    Generate a checksum for the dataset to verify reproducibility.
    
    Purpose:
        Creates a deterministic hash of the dataset contents that can be used to verify
        that the same exact dataset is produced on different runs.
        
    Args:
        df: DataFrame to check
        
    Returns:
        str: Deterministic MD5 hash checksum of the dataset
        
    Expected Value:
        The final verification checksum should be consistent across all environments
        when using the same input data and parameters.
    """
    
    # Sort the dataframe to ensure consistent ordering
    df_sorted = df.sort_values(by=['text']).reset_index(drop=True)
    
    # Create a deterministic string representation of the dataset
    dataset_str = ''.join(df_sorted['text'].astype(str).tolist())
    
    # Use MD5 hash for consistency across environments instead of built-in hash()
    checksum = hashlib.md5(dataset_str.encode('utf-8')).hexdigest()
    
    return checksum

def main() -> None:
    
    """
    Main execution function for the Persian sentiment dataset balancing pipeline.
    
    Purpose:
        Orchestrates the entire workflow from data loading through balancing,
        visualization, and saving, with strict controls for reproducibility.
        
    Workflow:
        1. Set all random seeds at the start
        2. Load cleaned dataset
        3. Analyze original distribution
        4. Create balanced dataset with reproducibility controls
        5. Analyze balanced distribution
        6. Verify reproducibility
        7. Visualize comparison
        8. Save balanced dataset
        9. Print comparison summary
    """
    
    logger.info("Starting Persian Sentiment Analysis - Reproducible Balanced Dataset Creation")
    
    # Step 1: Set all random seeds at the start for reproducibility
    print(f"\nUsing fixed random seed: {RANDOM_SEED} for reproducibility")
    set_all_seeds(RANDOM_SEED)
    
    # Step 2: Load the cleaned dataset
    df = load_dataset()
    if df is None:
        logger.error("Dataset loading failed. Exiting.")
        return
    
    # Step 3: Analyze the original dataset distribution
    original_counts = analyze_sentiment_distribution(df, "Original Dataset")
    original_total = len(df)
    
    # Step 4: Create a balanced dataset with reproducibility controls
    balanced_df = create_reproducible_balanced_dataset(df, sample_size=SAMPLE_SIZE)
    
    # Step 5: Analyze the balanced dataset distribution
    balanced_counts = analyze_sentiment_distribution(balanced_df, "Balanced Dataset")
    balanced_total = len(balanced_df)
    
    # Step 6: Verify reproducibility with a deterministic checksum
    checksum = verify_reproducibility(balanced_df)
    print(f"\nReproducibility verification checksum: {checksum}")
    
    # Validate verification checksum if expected value is provided
    if EXPECTED_VERIFICATION_CHECKSUM is not None:
        if checksum == EXPECTED_VERIFICATION_CHECKSUM:
            logger.info("Verification checksum validation: PASSED ✓")
            print("Verification checksum validation: PASSED ✓")
        else:
            logger.warning(f"Verification checksum validation: FAILED ✗ (Expected: {EXPECTED_VERIFICATION_CHECKSUM}, Got: {checksum})")
            print(f"Verification checksum validation: FAILED ✗ (Expected: {EXPECTED_VERIFICATION_CHECKSUM}, Got: {checksum})")
    
    print("This value should be identical across different runs with the same input file.")
    
    # Step 7: Visualize the comparison
    visualize_comparison(
        original_counts,
        balanced_counts,
        original_total,
        balanced_total
    )
    
    # Step 8: Save the balanced dataset
    save_success = save_balanced_dataset(balanced_df)
    
    # Step 9: Print comparison summary
    print_comparison_summary(
        original_counts,
        balanced_counts,
        original_total,
        balanced_total
    )
    
    # Final message
    if save_success:
        logger.info("Persian Sentiment Analysis - Reproducible Balanced Dataset Creation completed successfully")
        print("\nProcessing complete! The balanced dataset has been saved and downloaded.")
        print(f"Reproducibility checksum: {checksum}")
    else:
        logger.warning("Persian Sentiment Analysis - Reproducible Balanced Dataset Creation completed with warnings")
        print("\nProcessing complete with warnings. Please check the log for details.")

if __name__ == "__main__":
    main()
