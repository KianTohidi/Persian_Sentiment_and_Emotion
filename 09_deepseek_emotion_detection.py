"""
Persian Emotion Detection using DeepSeek-V3 model via API call

This script performs Persian emotion detection using the DeepSeek API and automates evaluation and visualization of results.
Input: A file containing 1800 balanced Persian texts with emotion labels (emotion_balanced.csv)
       (300 samples per emotion category)
Output: Multiple analysis files including DeepSeek_emotion_results.csv (main results),
        confusion matrix, classification reports, and visualization charts

Purpose:
    This module provides functionality to classify emotions in Persian text using the DeepSeek API.
    It processes datasets of Persian text, detects emotions using AI, evaluates classification
    performance, and generates analytical reports with visualizations to support emotion detection
    research.

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - requests (tested with version 2.32.3)
    - tqdm (tested with version 4.66.1)
    - scikit-learn (tested with version 1.4.1)
    - matplotlib (tested with version 3.10.0)
    - seaborn (tested with version 0.13.2)
"""

# IMPORTANT: Install dependencies before running this script
# !pip install requests pandas numpy tqdm scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import requests
import time
import json
import traceback
import io
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Optional, Union, Tuple, Any
from google.colab import files

# Set global pandas display format to 2 decimal places
pd.set_option('display.float_format', '{:.2f}'.format)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("emotion_detection.log")  # File output
    ]
)
logger = logging.getLogger(__name__)

# Constants
# API key should be loaded from environment variable or config file
# NEVER hardcode credentials in source code
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V3 model identifier
VALID_EMOTIONS = ['anger', 'fear', 'happiness', 'hate', 'sadness', 'surprise']
# Multiple encodings to try, in order of preference, for handling Persian text
ENCODING_OPTIONS = ['utf-8-sig', 'utf-8', 'windows-1256', 'cp1256', 'ISO-8859-6']
# File paths for all output files - centralized for easy modification
OUTPUT_FILES = {
    'results': "DeepSeek_emotion_results.csv",
    'temp_results': "DeepSeek_emotion_results_temp.csv",  # For atomic file operations
    'confusion': "DeepSeek_emotion_confusion_matrix.csv",
    'report': "DeepSeek_emotion_classification_report.csv",
    'heatmap': "DeepSeek_emotion_confusion_heatmap.png",
    'confusion_pairs': "DeepSeek_emotion_confusion_pairs.csv",
    'per_emotion': "DeepSeek_emotion_accuracy.csv"
}

# Note: This implementation is specifically optimized for Google Colab to make it
# accessible for beginner researchers. The file upload/download mechanisms and fixed
# output paths are intentionally designed for the Colab environment, which provides
# a consistent and beginner-friendly interface. For production environments, these
# would be parameterized for greater flexibility.

def get_api_key() -> str:
    """
    Get the DeepSeek API key from Colab secrets or manual input.

    Returns:
        str: The API key
    """
    # ======================================================================
    # HOW TO SET UP YOUR API KEY (2 SIMPLE OPTIONS)
    # ======================================================================
    # OPTION 1: RECOMMENDED - Using Colab Secrets (Secure & Persistent)
    #    - Click the 🔑 key icon in the left sidebar
    #    - Click "Add new secret"
    #    - For name: enter DEEPSEEK_API_KEY
    #    - For value: paste your actual API key
    #    - Click "Save"
    #
    # OPTION 2: Manual Input (Fallback)
    #    - If you don't set up a secret, you'll be prompted to
    #      enter your API key when the code runs
    #    - Note: You'll need to re-enter it each session
    # ======================================================================

    # First try to get the API key from Colab secrets
    # Note: This implementation prioritizes user experience in Google Colab,
    # which is widely used by NLP researchers for quick experimentation.
    # A production version would use a more flexible configuration approach.
    try:
        from google.colab import userdata
        api_key = userdata.get('DEEPSEEK_API_KEY')
        if api_key and api_key.strip():
            logger.info("✅ API key loaded from Colab secrets")
            return api_key
    except:
        # This executes if we're not in Colab or secrets aren't available
        logger.info("API key not found in Colab secrets")

    # As a fallback, prompt the user to enter the key manually
    logger.warning("⚠️ API key not found in Colab secrets")
    logger.warning("⚠️ For security, please add it to Colab secrets using the key icon in the sidebar")
    api_key = input("Enter your DeepSeek API key (will not be stored permanently): ")
    return api_key

def test_api_connection() -> bool:
    """
    Test if the DeepSeek API is working correctly.

    Purpose:
        Verifies the API connection to avoid running costly classification tasks
        if the API is unavailable or the API key is invalid.

    Input:
        None - Uses API key from environment variables.

    Output:
        Log messages indicating connection status.

    Returns:
        bool: True if the connection was successful, False otherwise.

    Raises:
        Exceptions from the requests library are caught and handled internally.
    """
    logger.info("Testing DeepSeek API connection...")
    try:
        # Get API key securely
        api_key = get_api_key()

        # Set up authentication and content headers for the API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Simple test payload that should result in a very short response
        # Uses a simple instruction to check if the API can respond successfully
        test_payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'DeepSeek API is working!' if you can receive this message."}
            ],
            "max_tokens": 20  # Limiting response size for this test
        }

        # Make the API call and check the response
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=test_payload)

        # Status code 200 indicates successful API communication
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"API test successful! Response: {content}")
            return True
        else:
            # Non-200 status codes indicate API issues (auth problems, rate limits, etc.)
            logger.error(f"API test failed. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        # Handle network errors, timeouts, or other unexpected issues
        logger.error(f"API test exception: {str(e)}")
        return False

def batch_detect_emotions(texts: List[str], model: str = DEFAULT_MODEL, max_retries: int = 3) -> Dict[str, str]:
    """
    Process multiple texts in a single API call to detect emotions.

    Purpose:
        Performs batch emotion detection on Persian texts using the DeepSeek API,
        reducing the number of API calls by grouping multiple inputs together.

    Input:
        texts: List of Persian text strings to analyze.
        model: DeepSeek model name to use. Defaults to DEFAULT_MODEL.
        max_retries: Maximum number of retry attempts on failure.

    Output:
        Dictionary mapping text IDs (as strings) to detected emotions.

    Returns:
        Dict[str, str]: A dictionary where keys are text IDs (starting from "1") and
                       values are emotion labels from the VALID_EMOTIONS list.

    Raises:
        JSONDecodeError: Handled internally if API returns invalid JSON.
        RequestException: Handled internally for network issues.
    """
    start_time = time.time()

    # Create a carefully designed prompt that includes clear instructions
    # The prompt engineering here is critical for accurate emotion detection
    prompt = """
You are a precise emotion classifier for Persian text. You must classify each text into EXACTLY ONE of these six emotions:
- anger
- fear
- happiness
- hate
- sadness
- surprise

IMPORTANT CONSTRAINTS:
1. Use ONLY these six emotions - no variations, no additional emotions
2. Return EXACTLY one emotion per text
3. Use lowercase for all emotion labels
4. Do not explain your choices or add any text besides the JSON object

Classify the emotion in each of these Persian texts:

"""

    # Add each text to the prompt with clear numbering for unambiguous identification
    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: {text}\n\n"

    # Add clear response format instructions to enforce proper output structure
    # This ensures we get consistent JSON that can be easily parsed
    prompt += """
Your response must be a valid JSON object with this exact format:
{
  "1": "emotion_name",
  "2": "emotion_name",
  ...
}

Where emotion_name is ONLY one of: anger, fear, happiness, hate, sadness, surprise.
"""

    # Initialize retries
    retries = 0

    while retries <= max_retries:
        try:
            # Get API key securely
            api_key = get_api_key()

            # Set up authentication headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Configure API call with parameters optimized for this task:
            # - temperature=0: Ensures deterministic results for consistent classification
            # - response_format as JSON: Forces structured output that's easy to parse
            # - System message sets context for accurate emotion analysis
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You analyze emotions in Persian texts and return results in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,  # Deterministic results are critical for consistent classification
                "max_tokens": 500,  # Sufficient for JSON responses containing many text classifications
                "response_format": {"type": "json_object"}  # Forces JSON output format
            }

            # Make the API call and check for successful response
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

            # Handle unsuccessful API responses
            if response.status_code != 200:
                error_message = f"API Error: Status code {response.status_code}"
                error_detail = f"Response: {response.text}"
                logger.error(error_message)
                logger.error(error_detail)
    
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to process batch after {max_retries} attempts. Skipping batch.")
                    # Return dictionary with error indication rather than empty dict
                    return {"error": error_message, "details": error_detail}       

            # Extract and parse the response data
            response_data = response.json()

            # Navigate the response structure to extract content
            # The API returns a specific structure we need to parse correctly
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "{}")

                # Parse JSON content - handle both string and dict formats
                # Some API versions might return pre-parsed JSON or JSON string
                if isinstance(content, str):
                    try:
                        results = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {content}")
                        logger.error(f"JSON parse error: {str(e)}")
                        if retries < max_retries:
                            retries += 1
                            wait_time = 2 ** retries  # Exponential backoff
                            logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            return {}  # Return empty dict after max retries
                else:
                    results = content
            else:
                # Handle unexpected API response format
                logger.error(f"Unexpected response format: {response_data}")
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}  # Return empty dict after max retries

            # Validate the emotion values in the results
            validated_results = {}
            for text_id, emotion in results.items():
                if emotion.lower() in VALID_EMOTIONS:
                    validated_results[text_id] = emotion.lower()
                else:
                    logger.warning(f"Invalid emotion '{emotion}' detected for text {text_id}. Skipping.")

            # Report timing information for monitoring performance
            elapsed_time = time.time() - start_time
            logger.info(f"✓ Processed {len(texts)} texts in {elapsed_time:.2f}s")
            return validated_results

        except Exception as e:
            # Handle any unexpected errors and show timing information
            elapsed_time = time.time() - start_time
            logger.error(f"× Error: {str(e)} (after {elapsed_time:.2f}s)")

            # Retry logic
            if retries < max_retries:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return {}

    return {}  # Fallback return if all retries fail

def read_csv_with_multiple_encodings(csv_content: Union[str, bytes]) -> pd.DataFrame:
    """
    Try multiple encodings to read a CSV file from content.

    Purpose:
        Handles encoding challenges common with Persian text files by
        systematically attempting multiple encodings until finding one that works.

    Input:
        csv_content: Either file path (str) or file content (bytes) to read.

    Output:
        pandas DataFrame containing the loaded CSV data with proper encoding.

    Returns:
        pd.DataFrame: The loaded CSV data.

    Raises:
        ValueError: If the CSV couldn't be read with any of the attempted encodings.
    """
    # Try each encoding in order until one works
    # Persian text files often have inconsistent encoding standards
    for encoding in ENCODING_OPTIONS:
        try:
            if isinstance(csv_content, bytes):
                # For uploaded files (bytes content from file upload)
                df = pd.read_csv(io.BytesIO(csv_content), encoding=encoding)
            else:
                # For local files (file path string)
                df = pd.read_csv(csv_content, encoding=encoding)
            logger.info(f"Successfully read with encoding: {encoding}")
            return df
        except Exception:
            # If this encoding fails, try the next one
            continue

    # If all encodings fail, raise an error
    raise ValueError("Could not read CSV with any of the attempted encodings")

def safe_save_dataframe(df: pd.DataFrame, filepath: str, temp_filepath: Optional[str] = None) -> bool:
    """
    Safely save a DataFrame to CSV using atomic file operations.

    Purpose:
        Prevents data loss by first writing to a temporary file and then
        renaming it, which is an atomic operation in most file systems.

    Input:
        df: DataFrame to save
        filepath: Final destination path
        temp_filepath: Temporary file path (generated if None)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create temporary filepath if not provided
        if temp_filepath is None:
            temp_filepath = f"{filepath}.tmp"

        # Save to temporary file first
        df.to_csv(temp_filepath, index=False, encoding='utf-8-sig')

        # Rename (atomic operation) to target file
        os.replace(temp_filepath, filepath)
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame: {str(e)}")
        return False

def process_dataset_batched(
    csv_content: Union[str, bytes],
    output_path: str,
    batch_size: int = 20,
    max_samples: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Process the dataset using batched API calls for efficiency.

    Purpose:
        Processes a dataset of Persian texts by sending them to the DeepSeek API
        in optimized batches, with progress tracking and incremental result saving.

    Input:
        csv_content: Either file path or uploaded file content containing Persian texts.
                    Expected CSV format: Must have a 'text' column with Persian texts.
        output_path: Path to save the results CSV file incrementally.
        batch_size: Number of texts to process in each API call (default: 20).
                   Higher values reduce API calls but increase failure risk.
        max_samples: Maximum number of samples to process (default: None = process all).

    Output:
        DataFrame with original data plus 'predicted_emotion' column.
        CSV file saved to output_path with the results.

    Returns:
        pd.DataFrame or None: Processed dataframe with predictions, or None if error occurred.

    Raises:
        ValueError: If the CSV is missing required columns.
        Various exceptions are caught and handled internally.
    """
    try:
        # Read dataset with proper encoding handling
        df = read_csv_with_multiple_encodings(csv_content)

        # Validate required columns exist in the dataset
        if "text" not in df.columns:
            raise ValueError("No 'text' column found in CSV file. The dataset must contain a 'text' column.")

        # Add emotion column if missing (for proper structure)
        if "emotion" not in df.columns:
            logger.warning("No 'emotion' column found in CSV file. Adding placeholder values.")
            df['emotion'] = "unknown"  # Add placeholder for evaluation structure

        # Sample data if requested (for testing or limited processing)
        if max_samples and max_samples < len(df):
            # Use fixed random seed for reproducible sampling
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from the dataset.")

        # Initialize prediction column if it doesn't exist
        # This also supports resuming interrupted processing
        if 'predicted_emotion' not in df.columns:
            df['predicted_emotion'] = None

        # Process only unpredicted rows to support incremental processing
        # This is crucial for handling large datasets or recovering from failures
        unpredicted_df = df[df['predicted_emotion'].isnull()].copy()
        total_to_process = len(unpredicted_df)
        processed_count = 0

        logger.info(f"Found {total_to_process} rows to process")

        # Track overall processing time for accurate ETA calculation
        overall_start = time.time()

        # Process in batches with progress bar for monitoring
        for i in tqdm(range(0, len(unpredicted_df), batch_size)):
            # Extract the current batch, ensuring we don't go past the end
            batch = unpredicted_df.iloc[i:min(i+batch_size, len(unpredicted_df))]
            if batch.empty:
                continue

            # Get batch predictions using the emotion detection function
            emotion_results = batch_detect_emotions(batch['text'].tolist())

            # Update dataframe with results if we got valid predictions
            if emotion_results:
                for idx, row in batch.iterrows():
                    # Find the position of this row within the current batch
                    batch_position = batch.index.get_loc(idx) + 1  # +1 because API indexing starts at 1
                    text_id = str(batch_position)  # Map to the ID format used in batch_detect_emotions

                    if text_id in emotion_results:
                        # Store the prediction in the main dataframe
                        df.at[idx, 'predicted_emotion'] = emotion_results[text_id]
                        processed_count += 1

            # Save progress after each batch to handle interruptions gracefully
            # Use atomic file operations to prevent data corruption
            safe_save_dataframe(df, output_path, OUTPUT_FILES['temp_results'])

            # Calculate and display estimated time remaining for user feedback
            if processed_count > 0:
                elapsed = time.time() - overall_start
                # Calculate average processing time per batch
                avg_time_per_batch = elapsed / (i + min(batch_size, len(unpredicted_df) - i)) * batch_size
                # Estimate remaining batches
                remaining_batches = (total_to_process - processed_count) / batch_size
                # Calculate estimated time remaining
                est_remaining = avg_time_per_batch * remaining_batches
                # Display progress with time estimate
                logger.info(f"Saved progress. Batch {i//batch_size + 1} complete. Estimated remaining time: {datetime.timedelta(seconds=int(est_remaining))}")

            # Small delay between batches to avoid API rate limiting
            time.sleep(1)  # Brief pause between batches

        # Display final statistics for the entire processing job
        total_time = time.time() - overall_start
        logger.info(f"Total processing time: {datetime.timedelta(seconds=int(total_time))}")
        logger.info(f"Processed {processed_count} out of {total_to_process} rows")

        # For backward compatibility - show processing time in standard format
        print(f"Total processing time: {datetime.timedelta(seconds=int(total_time))}")

        return df

    except Exception as e:
        # Comprehensive error handling with traceback for debugging
        logger.error(f"Error processing dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def evaluate_results(df: pd.DataFrame) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
    """
    Evaluate the performance of the emotion detection model.

    Purpose:
        Generates comprehensive metrics and visualizations for emotion classification
        results, including accuracy, confusion matrix, and per-emotion performance analysis.

    Input:
        df: DataFrame with 'emotion' (ground truth) and 'predicted_emotion' columns.
           Expected to contain balanced emotion labels for statistical validity.

    Output:
        Console output with classification performance metrics.
        Multiple output files:
          - Confusion matrix CSV
          - Classification report CSV
          - Confusion heatmap visualization
          - Commonly confused emotion pairs analysis
          - Per-emotion accuracy breakdown

    Returns:
        Tuple containing:
            - accuracy (float or None): Overall classification accuracy, or None if evaluation failed
            - confusion_matrix (DataFrame or None): Confusion matrix as DataFrame, or None if evaluation failed

    Raises:
        Various exceptions are caught and handled internally.
    """
    # Validate input data before proceeding
    if df is None or df.empty or df['predicted_emotion'].isna().all():
        logger.error("No predictions were made. Evaluation cannot proceed.")
        return None, None

    # Filter to only include predictions with valid emotion labels
    # This prevents errors from invalid labels in the evaluation metrics
    df_valid = df[df['predicted_emotion'].isin(VALID_EMOTIONS)]

    if len(df_valid) == 0:
        logger.error("No valid emotion predictions found. Evaluation cannot proceed.")
        return None, None

    try:
        # Calculate overall accuracy (primary performance metric)
        accuracy = (df_valid['emotion'] == df_valid['predicted_emotion']).mean()
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        # Mirror to standard output for compatibility
        print(f"Overall accuracy: {accuracy:.4f}")

        # Generate detailed classification report with precision, recall, and F1-score
        logger.info("\nClassification Report:")
        cr = classification_report(df_valid['emotion'], df_valid['predicted_emotion'], output_dict=True)

        # Round numeric values to 2 decimal places for readability
        for emotion in cr:
            if isinstance(cr[emotion], dict):
                for metric in cr[emotion]:
                    if isinstance(cr[emotion][metric], float):
                        cr[emotion][metric] = round(cr[emotion][metric], 2)

        # Convert to DataFrame for easier viewing and export
        cr_df = pd.DataFrame(cr).transpose()
        # Set pandas display options to consistently show 2 decimal places
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(cr_df)  # Keep print for data display


        # Save classification report to CSV for further analysis
        cr_df.to_csv(OUTPUT_FILES['report'])
        files.download(OUTPUT_FILES['report'])

        # Generate confusion matrix to analyze error patterns
        # The confusion matrix shows where the model makes specific mistakes
        cm = confusion_matrix(
            df_valid['emotion'],
            df_valid['predicted_emotion'],
            labels=VALID_EMOTIONS  # Ensure consistent ordering
        )
        # Convert to DataFrame with proper row/column labels
        cm_df = pd.DataFrame(cm, index=VALID_EMOTIONS, columns=VALID_EMOTIONS)
        print("\nConfusion Matrix:")  # Keep print for data display
        print(cm_df)

        # Save confusion matrix to CSV for further analysis
        cm_df.to_csv(OUTPUT_FILES['confusion'])
        files.download(OUTPUT_FILES['confusion'])

        # Create and save confusion matrix heatmap visualization
        # Visual representation helps identify patterns in the errors
        plt.figure(figsize=(10, 8))
        # Blues colormap provides good contrast for the confusion matrix
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Emotion Classification Confusion Matrix")
        plt.ylabel("True Emotion")
        plt.xlabel("Predicted Emotion")
        plt.tight_layout()
        plt.savefig(OUTPUT_FILES['heatmap'], dpi=300, bbox_inches='tight')
        files.download(OUTPUT_FILES['heatmap'])

        # Identify and analyze commonly confused emotion pairs
        # This helps understand which emotions are most difficult to distinguish
        mistakes = df_valid[df_valid['emotion'] != df_valid['predicted_emotion']]
        if len(mistakes) > 0:
            print("\nCommonly confused pairs:")  # Keep print for data display
            # Group by true and predicted emotion to find common error patterns
            confusion_pairs = mistakes.groupby(['emotion', 'predicted_emotion']).size().reset_index(name='count')
            # Sort by frequency to highlight the most common confusions
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
            print(confusion_pairs.head(10))  # Show top 10 confusion pairs

            # Save confusion pairs to CSV for detailed analysis
            confusion_pairs.to_csv(OUTPUT_FILES['confusion_pairs'], index=False)
            files.download(OUTPUT_FILES['confusion_pairs'])

        # Calculate per-emotion accuracy to identify problematic categories
        per_emotion_accuracy = []
        print("\nPer-emotion accuracy:")  # Keep print for data display
        for emotion in VALID_EMOTIONS:
            # Filter for each emotion category
            emotion_subset = df_valid[df_valid['emotion'] == emotion]
            if len(emotion_subset) > 0:
                # Calculate accuracy for this specific emotion
                emotion_acc = (emotion_subset['predicted_emotion'] == emotion).mean()
                per_emotion_accuracy.append({
                    'emotion': emotion,
                    'accuracy': emotion_acc,
                    'sample_count': len(emotion_subset)
                })
                print(f"{emotion}: {emotion_acc:.4f} (n={len(emotion_subset)})")

        # Save per-emotion accuracy metrics
        per_emotion_df = pd.DataFrame(per_emotion_accuracy)
        per_emotion_df.to_csv(OUTPUT_FILES['per_emotion'], index=False)
        files.download(OUTPUT_FILES['per_emotion'])

        # Create and display per-emotion accuracy chart for visual comparison
        # This visualization helps identify which emotions are hardest to detect
        # Note: Using fixed visualization settings for consistency and simplicity.
        # This could be parameterized in future versions for more flexibility.
        plt.figure(figsize=(10, 6))
        # Use proper hue parameter setting (fixing FutureWarning)
        sns.barplot(x='emotion', y='accuracy', hue='emotion', data=per_emotion_df,
            palette='viridis', legend=False)
        plt.title('Accuracy by Emotion Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Emotion')
        plt.ylim(0, 1)  # Consistent scale for accuracy
        plt.xticks(rotation=45)  # Rotate labels for readability
        plt.tight_layout()
        plt.savefig('DeepSeek_emotion_accuracy_chart.png', dpi=300, bbox_inches='tight')
        files.download('DeepSeek_emotion_accuracy_chart.png')

        print("\nInitiating downloads for evaluation results...")
        print(f"Download links created for evaluation files")

        return accuracy, cm_df

    except Exception as e:
        # Handle errors during evaluation
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    """
    Main execution function for the emotion classification pipeline.

    Purpose:
        Orchestrates the entire workflow from API testing through data processing
        to evaluation and results presentation. Functions as the entry point for
        the program's execution.

    Workflow:
        1. Test API connection (via test_api_connection)
        2. Upload CSV file (via Colab interface)
        3. Process dataset in batches (via process_dataset_batched)
        4. Download results file
        5. Evaluate model performance (via evaluate_results)
        6. Generate and download evaluation files

    Input:
        None directly. User will upload a CSV file through the Colab interface.
        Expected file format: CSV with 'text' and 'emotion' columns, containing
        Persian text samples and ground truth emotion labels.

    Output:
        Multiple files with analysis results (saved to OUTPUT_FILES locations).
        Console output with progress updates and performance metrics.
    """
    # Test API connection before proceeding with the full workflow
    # This prevents wasting time if the API is unavailable
    if not test_api_connection():
        logger.error("Failed to connect to DeepSeek API. Please check your API key and internet connection.")
        return

    # Upload file through Google Colab interface
    print("Please upload emotion_balanced.csv file:")  # Keep print for user interaction
    uploaded = files.upload()

    # Extract the filename and content of the uploaded file
    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # Process dataset with the uploaded file content
    # Use batch_size=20 which was determined to be optimal through testing:
    # - Small enough to avoid timeouts
    # - Large enough to efficiently use API quota
    results_df = process_dataset_batched(
        file_content,
        OUTPUT_FILES['results'],
        batch_size=20,  # Optimal batch size determined through testing
        max_samples=None  # Process all samples in the dataset
    )

    # Auto-download the results file for immediate access
    print("\nInitiating download for main results file...")  # Keep print for user interaction
    files.download(OUTPUT_FILES['results'])
    print(f"Download link created for {OUTPUT_FILES['results']}")

    # Evaluate results if processing was successful
    if results_df is not None and not results_df.empty:
        # Generate comprehensive evaluation metrics and visualizations
        accuracy, confusion = evaluate_results(results_df)

        # Final summary message with file listing
        print("\nProcessing complete! All results have been saved and download links have been created.")
        print("Please check your downloads folder for the following files:")
        for file_type, file_name in OUTPUT_FILES.items():
            if 'temp' not in file_type:  # Don't show temporary files
                print(f"- {file_name}")
        print("- DeepSeek_emotion_accuracy_chart.png")
    else:
        print("No results to evaluate. Please check for errors above.")

    # TODO: Future improvements:
    # 1. Add unit tests to verify the core functionality
    # 2. Create a configuration file for customizable parameters
    # 3. Adapt code to work in non-Colab environments while maintaining simplicity
    # 4. Add more customization options for visualizations

if __name__ == "__main__":
    main()
