"""
Persian Emotion Detection - Reproducible Balanced Dataset Creation
------------------------------------------------------------------

This script creates a balanced dataset for Persian emotion detection by sampling an equal 
number of entries from each emotion class with GUARANTEED reproducibility.

Purpose:
    Creates a balanced dataset from an imbalanced emotion-labeled dataset by sampling equal
    numbers from each emotion class (anger, fear, happiness, hate, sadness, surprise), with 
    strict controls to ensure exact reproducibility across different runs and environments.

### IMPORTANT: Before running this code:
    1. Make sure you have already run the dataset cleaning script
    2. Have the cleaned CSV file ready for upload (emotion_ecommerce_cleaned.csv)

Input:
    Cleaned CSV file with Persian texts and emotion labels (emotion_ecommerce_cleaned.csv)
    Expected format: CSV with 'text' and 'emotion' columns (emotion values: 'anger', 'fear', 
                     'happiness', 'hate', 'sadness', 'surprise')

Output:
    - "emotion_balanced.csv": Reproducible balanced dataset with equal representation of each emotion class
    - Statistical reports comparing original and balanced distributions
    - Percentage change analysis between original and balanced datasets

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - matplotlib (optional, for visualization)
    - google.colab (for file upload/download in Colab environment)
    
Expected Checksum Values:
    After first running this code successfully, update these values in the constants section:
    - Dataset creation checksum: Will be generated on first run
    - Final verification checksum: Will be generated on first run
"""

"""
Reproducibility Guarantee and Limitations
-----------------------------------------
This code is designed to guarantee identical output every time, regardless of environment or user, as long as:
- The input file is the same
- The sample size parameter is the same
- The defined emotion classes remain the same

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
import hashlib  # For deterministic checksums
from typing import Dict, Tuple, Optional, Union, List
from google.colab import files
import matplotlib.pyplot as plt

# Configure logging for detailed process tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("emotion_balancing.log")  # File output
    ]
)
logger = logging.getLogger(__name__)

# Constants for consistent reference
INPUT_FILE = "emotion_ecommerce_cleaned.csv"
OUTPUT_FILE = "emotion_balanced.csv"
ENCODING = "utf-8-sig"  # Standard encoding for Persian text
RANDOM_SEED = 42  # Fixed seed for reproducibility
SAMPLE_SIZE = 300  # Default number of entries to sample from each emotion class
# Updated emotion classes in alphabetical order
EMOTION_CLASSES = ['anger', 'fear', 'happiness', 'hate', 'sadness', 'surprise']  # Expected emotion classes in alphabetical order

# Checksum validation constants
# These will be MD5 hash strings
EXPECTED_CREATION_CHECKSUM = "dd6d3b0c1f2deea0922da888d3605539"
EXPECTED_VERIFICATION_CHECKSUM = "e4840cd8fa1bcbee9ecca414392d3d4e"

def set_all_seeds(seed_value: int) -> None:
    
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
    Load the cleaned emotion dataset from a CSV file with proper encoding for Persian text.
    
    Purpose:
        Handles file upload in Colab environment and loads the dataset with the
        appropriate encoding for Persian text.
        
    Returns:
        pd.DataFrame or None: Loaded dataset if successful, None otherwise
    """
    
    logger.info("Starting dataset load process")
    try:
        # Handle file upload in Colab
        logger.info(f"Prompting user to upload cleaned emotion dataset")
        print(f"Please upload your emotion ecommerce cleaned CSV file: ({INPUT_FILE})")
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

def analyze_emotion_distribution(df: pd.DataFrame, title: str = "Dataset") -> Dict[str, int]:
    
    """
    Analyze and display the emotion distribution in the dataset.
    
    Purpose:
        Calculates and displays the distribution of emotion labels in the dataset,
        including counts and percentages for each emotion class.
        
    Args:
        df: DataFrame containing an 'emotion' column
        title: Description of the dataset for display purposes
        
    Returns:
        Dict[str, int]: Dictionary of emotion counts keyed by emotion label
    """
    
    logger.info(f"Analyzing emotion distribution for: {title}")
    
    # Calculate emotion counts
    emotion_counts = df['emotion'].value_counts()
    total = len(df)
    
    # Display information
    print(f"\n=== {title.upper()} INFORMATION ===")
    print(f"Total number of entries: {total}")
    
    if title == "Original Dataset":
        print(f"Columns: {', '.join(df.columns)}")
    
    # Show the distribution of emotion labels
    print("\nEmotion Distribution:")
    # Use a consistent order for emotion classes to ensure reproducible output
    counts_dict = emotion_counts.to_dict()
    for emotion in EMOTION_CLASSES:
        count = counts_dict.get(emotion, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{emotion.capitalize()}: {count} entries ({percentage:.1f}%)")
    
    return counts_dict

def create_reproducible_balanced_dataset(df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    
    """
    Create a balanced dataset with guaranteed reproducibility.
    
    Purpose:
        Ensures equal representation of each emotion class by sampling a specified
        number of entries from each class, with strict controls for reproducibility.
        
    Args:
        df: DataFrame containing text and emotion columns
        sample_size: Number of entries to sample from each emotion class
        
    Returns:
        pd.DataFrame: Balanced dataset with equal representation of each emotion class
        
    Expected Checksum:
        Will generate a deterministic MD5 hash that should be consistent across all
        environments when using the same input data and parameters.
    """
    
    logger.info(f"Creating balanced dataset with {sample_size} samples per emotion class")
    
    # Create a fresh DataFrame to hold our samples
    sampled_data = pd.DataFrame(columns=df.columns)
    
    # Reset the random seed before starting any sampling
    # This ensures the sampling process always starts from the same state
    set_all_seeds(RANDOM_SEED)
    
    # Process each emotion category in a consistent order
    for emotion in EMOTION_CLASSES:
        # Filter the dataframe for the current emotion
        emotion_df = df[df['emotion'] == emotion].copy()
        
        # Sort the filtered data by a consistent criterion to ensure deterministic ordering
        # This step is critical for reproducibility - we first set a deterministic order
        # before applying random sampling
        emotion_df = emotion_df.sort_values(by=['text']).reset_index(drop=True)
        
        # Check if we have enough data for this emotion
        if len(emotion_df) < sample_size:
            logger.warning(f"Not enough data for '{emotion}'. Only {len(emotion_df)} entries available.")
            print(f"\nWarning: Not enough data for '{emotion}'. Only {len(emotion_df)} entries available.")
            # Take all available data for this emotion
            sampled_emotion = emotion_df
        else:
            # Reset the random seed immediately before sampling
            # This step ensures each emotion class sampling starts from the same seed state
            np.random.seed(RANDOM_SEED)
            
            # Randomly sample the specified number of entries
            logger.info(f"Sampling {sample_size} entries for '{emotion}' emotion")
            sampled_emotion = emotion_df.sample(n=sample_size, random_state=RANDOM_SEED)
        
        # Add to the sampled dataframe
        sampled_data = pd.concat([sampled_data, sampled_emotion])
    
    # For the final shuffle, reset the random seed once more to ensure reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Sort by text first to establish a deterministic order before shuffling
    sampled_data = sampled_data.sort_values(by=['text']).reset_index(drop=True)
    
    # Now shuffle with the reset random seed for reproducible randomization
    logger.info("Shuffling the balanced dataset in a reproducible manner")
    sampled_data = sampled_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Add a deterministic checksum to verify dataset reproducibility
    # Using MD5 hash for consistency across environments
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
        Creates a bar chart comparing the emotion distributions in the
        original and balanced datasets as percentages.
        
    Args:
        original_counts: Dictionary of emotion counts in the original dataset
        balanced_counts: Dictionary of emotion counts in the balanced dataset
        original_total: Total number of entries in the original dataset
        balanced_total: Total number of entries in the balanced dataset
    """
                             
    logger.info("Creating comparison visualization for original and balanced datasets")
    
    # Reset the random seed before plotting for consistent colors/styles
    np.random.seed(RANDOM_SEED)
    
    # Get list of all emotion labels (ensuring consistent order)
    emotions = EMOTION_CLASSES
    
    # Calculate percentages
    original_percentages = [(original_counts.get(e, 0) / original_total) * 100 if original_total > 0 else 0 for e in emotions]
    balanced_percentages = [(balanced_counts.get(e, 0) / balanced_total) * 100 if balanced_total > 0 else 0 for e in emotions]
    
    # Set up the bar chart positions
    x = np.arange(len(emotions))
    width = 0.35
    
    # Create the bars
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, original_percentages, width, label='Original Dataset (%)', color='skyblue')
    rects2 = ax.bar(x + width/2, balanced_percentages, width, label='Balanced Dataset (%)', color='lightgreen')
    
    # Add labels and title
    ax.set_xlabel('Emotion', fontsize=14)
    ax.set_ylabel('Percentage', fontsize=14)
    ax.set_title('Emotion Distribution Comparison: Original vs Balanced (%)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions])
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
        original_counts: Dictionary of emotion counts in the original dataset
        balanced_counts: Dictionary of emotion counts in the balanced dataset
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
    for emotion in EMOTION_CLASSES:
        original_percentage = (original_counts.get(emotion, 0) / original_total) * 100 if original_total > 0 else 0
        balanced_percentage = (balanced_counts.get(emotion, 0) / balanced_total) * 100 if balanced_total > 0 else 0
        change = balanced_percentage - original_percentage
        print(f"{emotion.capitalize()}: {change:+.1f} percentage points")

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
    
    # Use MD5 hash for consistency across environments
    checksum = hashlib.md5(dataset_str.encode('utf-8')).hexdigest()
    
    return checksum

def main() -> None:
    
    """
    Main execution function for the Persian emotion dataset balancing pipeline.
    
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
    
    logger.info("Starting Persian Emotion Detection - Reproducible Balanced Dataset Creation")
    
    # Step 1: Set all random seeds at the start for reproducibility
    print(f"\nUsing fixed random seed: {RANDOM_SEED} for reproducibility")
    set_all_seeds(RANDOM_SEED)
    
    # Step 2: Load the cleaned dataset
    df = load_dataset()
    if df is None:
        logger.error("Dataset loading failed. Exiting.")
        return
    
    # Step 3: Analyze the original dataset distribution
    original_counts = analyze_emotion_distribution(df, "Original Dataset")
    original_total = len(df)
    
    # Step 4: Create a balanced dataset with reproducibility controls
    balanced_df = create_reproducible_balanced_dataset(df, sample_size=SAMPLE_SIZE)
    
    # Step 5: Analyze the balanced dataset distribution
    balanced_counts = analyze_emotion_distribution(balanced_df, "Balanced Dataset")
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
    
    # Step 7: Visualize the comparison (optional)
    try:
        visualize_comparison(
            original_counts,
            balanced_counts,
            original_total,
            balanced_total
        )
    except Exception as e:
        logger.warning(f"Visualization skipped: {str(e)}")
        print("Visualization skipped. If you need visualization, ensure matplotlib is installed.")
    
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
        logger.info("Persian Emotion Detection - Reproducible Balanced Dataset Creation completed successfully")
        print("\nProcessing complete! The balanced dataset has been saved and downloaded.")
        print(f"Reproducibility checksum: {checksum}")
    else:
        logger.warning("Persian Emotion Detection - Reproducible Balanced Dataset Creation completed with warnings")
        print("\nProcessing complete with warnings. Please check the log for details.")

if __name__ == "__main__":
    main()
