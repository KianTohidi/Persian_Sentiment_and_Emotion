"""
Persian Emotion Detection using Anthropic Claude 3.7 Sonnet model via API

This script performs Persian emotion detection using the Anthropic Claude API and automates evaluation and visualization of results.
Input: A file containing 1800 balanced Persian texts with emotion labels (emotion_balanced.csv)
       (300 samples per emotion category)
Output: Multiple analysis files including Claude_emotion_results.csv (main results),
        confusion matrix, classification reports, and visualization charts

Purpose:
    This module provides functionality to classify emotions in Persian text using the Anthropic Claude API.
    It processes datasets of Persian text, detects emotions using AI, evaluates classification
    performance, and generates analytical reports with visualizations to support emotion detection
    research.

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - anthropic (tested with version 0.25.0) 
    - tqdm (tested with version 4.66.1)
    - scikit-learn (tested with version 1.4.1)
    - matplotlib (tested with version 3.10.0)
    - seaborn (tested with version 0.13.2)
"""

# IMPORTANT: Install dependencies before running this script
# !pip install -q -U anthropic
# !pip install pandas numpy tqdm scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import anthropic
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
        logging.StreamHandler(),
        logging.FileHandler("emotion_detection.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
# API key should be loaded from environment variable or config file
# NEVER hardcode credentials in source code
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"  # Claude 3.7 Sonnet model identifier
VALID_EMOTIONS = ['anger', 'fear', 'happiness', 'hate', 'sadness', 'surprise']
# Multiple encodings to try, in order of preference, for handling Persian text
ENCODING_OPTIONS = ['utf-8-sig', 'utf-8', 'windows-1256', 'cp1256', 'ISO-8859-6']
# File paths for all output files - centralized for easy modification
OUTPUT_FILES = {
    'results': "Claude_emotion_results.csv",
    'temp_results': "Claude_emotion_results_temp.csv",  # For atomic file operations
    'confusion': "Claude_emotion_confusion_matrix.csv",
    'report': "Claude_emotion_classification_report.csv",
    'heatmap': "Claude_emotion_confusion_heatmap.png",
    'confusion_pairs': "Claude_emotion_confusion_pairs.csv",
    'per_emotion': "Claude_emotion_accuracy.csv"
}

# Note: This implementation is specifically optimized for Google Colab to make it
# accessible for beginner researchers. The file upload/download mechanisms and fixed
# output paths are intentionally designed for the Colab environment, which provides
# a consistent and beginner-friendly interface. For production environments, these
# would be parameterized for greater flexibility.

def get_api_key() -> str:
    """
    Get the Anthropic Claude API key from Colab secrets or manual input.

    Returns:
        str: The API key
    """
    # ======================================================================
    # HOW TO SET UP YOUR API KEY (2 SIMPLE OPTIONS)
    # ======================================================================
    # OPTION 1: RECOMMENDED - Using Colab Secrets (Secure & Persistent)
    #    - Click the 🔑 key icon in the left sidebar
    #    - Click "Add new secret"
    #    - For name: enter ANTHROPIC_API_KEY
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
        api_key = userdata.get('ANTHROPIC_API_KEY')
        if api_key and api_key.strip():
            logger.info("✅ API key loaded from Colab secrets")
            return api_key
    except:
        logger.info("API key not found in Colab secrets")

    logger.warning("⚠️ API key not found in Colab secrets")
    logger.warning("⚠️ For security, please add it to Colab secrets using the key icon in the sidebar")
    api_key = input("Enter your Anthropic Claude API key (will not be stored permanently): ")
    return api_key

def initialize_claude_client():
    """Initialize the Claude API client with the API key."""
    try:
        api_key = get_api_key()
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("✅ Claude API client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Claude API client: {str(e)}")
        return None

def test_api_connection() -> bool:
    """Test the connection with the Claude API."""
    logger.info("Testing Anthropic Claude API connection...")
    try:
        client = initialize_claude_client()
        if not client:
            return False
            
        # Send a simple test message to verify API connectivity
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=100,
            temperature=0,
            messages=[
                {"role": "user", "content": "Say 'Claude API is working!' if you can receive this message."}
            ]
        )

        if response and hasattr(response, 'content') and len(response.content) > 0:
            logger.info(f"API test successful! Response: {response.content[0].text}")
            return True
        else:
            logger.error("API test failed. Empty or invalid response.")
            return False
    except Exception as e:
        logger.error(f"API test exception: {str(e)}")
        return False

def batch_detect_emotions(texts: List[str], model_name: str = DEFAULT_MODEL, max_retries: int = 3) -> Dict[str, str]:
    """
    Detect emotions in a batch of Persian texts using Claude API.
    
    Args:
        texts: List of Persian text strings to analyze
        model_name: The Claude model to use
        max_retries: Maximum number of retry attempts for API calls
        
    Returns:
        Dictionary mapping text IDs to detected emotions
    """
    start_time = time.time()

    # Construct the prompt for the Claude model with clear instructions
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

    # Add each text to the prompt with a unique identifier
    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: {text}\n\n"

    # Specify the exact expected response format
    prompt += """
Your response must be a valid JSON object with this exact format:
{
  "1": "emotion_name",
  "2": "emotion_name",
  ...
}

Where emotion_name is ONLY one of: anger, fear, happiness, hate, sadness, surprise.
"""

    retries = 0
    client = None

    # Implement retry mechanism for API call robustness
    while retries <= max_retries:
        try:
            # Check if we need to reinitialize the client
            if client is None:
                client = initialize_claude_client()
                if client is None:
                    logger.error("Failed to initialize Claude client")
                    if retries < max_retries:
                        retries += 1
                        wait_time = 2 ** retries  # Exponential backoff strategy
                        logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {}

            # Make the actual API call to Claude
            response = client.messages.create(
                model=model_name,
                max_tokens=500,
                temperature=0,  # Use deterministic outputs for classification
                system="You are a Persian emotion classifier. You will return JSON only.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Validate the response
            if not response or not hasattr(response, 'content') or len(response.content) == 0:
                logger.error("Empty or invalid response from Claude API")
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}

            content = response.content[0].text

            try:
                # Extract the JSON part of the response
                # Claude sometimes adds explanatory text even when asked not to
                # This regex search finds JSON objects in the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    results = json.loads(json_str)
                else:
                    # If no JSON pattern found, try the whole response
                    results = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {content}")
                logger.error(f"JSON parse error: {str(e)}")
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}

            # Validate that all emotions are from our predefined list
            validated_results = {}
            for text_id, emotion in results.items():
                if emotion.lower() in VALID_EMOTIONS:
                    validated_results[text_id] = emotion.lower()
                else:
                    logger.warning(f"Invalid emotion '{emotion}' detected for text {text_id}. Skipping.")

            elapsed_time = time.time() - start_time
            logger.info(f"✓ Processed {len(texts)} texts in {elapsed_time:.2f}s")
            return validated_results

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"× Error: {str(e)} (after {elapsed_time:.2f}s)")

            if retries < max_retries:
                retries += 1
                wait_time = 2 ** retries
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return {}

    return {}

def read_csv_with_multiple_encodings(csv_content: Union[str, bytes]) -> pd.DataFrame:
    """
    Attempt to read a CSV file with multiple encodings until successful.
    
    Args:
        csv_content: Raw CSV content as string or bytes
        
    Returns:
        DataFrame containing the CSV data
        
    Raises:
        ValueError: If no encoding could successfully read the CSV
    """
    # Try multiple encodings to handle Persian text properly - this is crucial
    # for working with non-Latin scripts that may be encoded in various ways
    for encoding in ENCODING_OPTIONS:
        try:
            if isinstance(csv_content, bytes):
                df = pd.read_csv(io.BytesIO(csv_content), encoding=encoding)
            else:
                df = pd.read_csv(csv_content, encoding=encoding)
            logger.info(f"Successfully read with encoding: {encoding}")
            return df
        except Exception:
            continue

    raise ValueError("Could not read CSV with any of the attempted encodings")

def safe_save_dataframe(df: pd.DataFrame, filepath: str, temp_filepath: Optional[str] = None) -> bool:
    """
    Safely save a DataFrame to CSV using atomic file operations.
    
    Args:
        df: DataFrame to save
        filepath: Destination file path
        temp_filepath: Temporary file path for atomic operation
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Using atomic write operations to prevent data corruption if interrupted
        if temp_filepath is None:
            temp_filepath = f"{filepath}.tmp"

        # First write to a temporary file
        df.to_csv(temp_filepath, index=False, encoding='utf-8-sig')

        # Then atomically replace the destination file
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
    Process a dataset in batches, detecting emotions using Claude API.
    
    Args:
        csv_content: Raw CSV content as string or bytes
        output_path: Path to save the results
        batch_size: Number of texts to process in each batch
        max_samples: Maximum number of samples to process
        
    Returns:
        DataFrame with the processed results, or None on failure
    """
    try:
        # Load and prepare the dataset
        df = read_csv_with_multiple_encodings(csv_content)

        # Validate the required columns exist
        if "text" not in df.columns:
            raise ValueError("No 'text' column found in CSV file. The dataset must contain a 'text' column.")

        if "emotion" not in df.columns:
            logger.warning("No 'emotion' column found in CSV file. Adding placeholder values.")
            df['emotion'] = "unknown"

        # Optionally limit dataset size for testing or budget constraints
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from the dataset.")

        # Initialize prediction column if not present
        if 'predicted_emotion' not in df.columns:
            df['predicted_emotion'] = None

        # Filter to only process rows without predictions yet
        unpredicted_df = df[df['predicted_emotion'].isnull()].copy()
        total_to_process = len(unpredicted_df)
        processed_count = 0

        logger.info(f"Found {total_to_process} rows to process")

        overall_start = time.time()

        # Process in batches to optimize API usage and handle large datasets
        for i in tqdm(range(0, len(unpredicted_df), batch_size)):
            batch = unpredicted_df.iloc[i:min(i+batch_size, len(unpredicted_df))]
            if batch.empty:
                continue

            # Send batch to Claude API for emotion detection
            emotion_results = batch_detect_emotions(batch['text'].tolist())

            # Update the main dataframe with predictions
            if emotion_results:
                for idx, row in batch.iterrows():
                    batch_position = batch.index.get_loc(idx) + 1
                    text_id = str(batch_position)

                    if text_id in emotion_results:
                        df.at[idx, 'predicted_emotion'] = emotion_results[text_id]
                        processed_count += 1

            # Save progress after each batch for resilience against failures
            safe_save_dataframe(df, output_path, OUTPUT_FILES['temp_results'])

            # Show progress and estimated completion time
            if processed_count > 0:
                elapsed = time.time() - overall_start
                avg_time_per_batch = elapsed / (i + min(batch_size, len(unpredicted_df) - i)) * batch_size
                remaining_batches = (total_to_process - processed_count) / batch_size
                est_remaining = avg_time_per_batch * remaining_batches
                logger.info(f"Saved progress. Batch {i//batch_size + 1} complete. Estimated remaining time: {datetime.timedelta(seconds=int(est_remaining))}")

            # Avoid rate limiting
            time.sleep(1)

        # Log completion statistics
        total_time = time.time() - overall_start
        logger.info(f"Total processing time: {datetime.timedelta(seconds=int(total_time))}")
        logger.info(f"Processed {processed_count} out of {total_to_process} rows")

        print(f"Total processing time: {datetime.timedelta(seconds=int(total_time))}")

        return df

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def evaluate_results(df: pd.DataFrame) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
    """
    Evaluate the emotion detection results and generate reports and visualizations.
    
    Args:
        df: DataFrame containing the prediction results
        
    Returns:
        Tuple of (accuracy, confusion matrix DataFrame)
    """
    # Validate that we have data to evaluate
    if df is None or df.empty or df['predicted_emotion'].isna().all():
        logger.error("No predictions were made. Evaluation cannot proceed.")
        return None, None

    # Filter to only include valid emotion predictions
    df_valid = df[df['predicted_emotion'].isin(VALID_EMOTIONS)]

    if len(df_valid) == 0:
        logger.error("No valid emotion predictions found. Evaluation cannot proceed.")
        return None, None

    try:
        # Calculate overall accuracy
        accuracy = (df_valid['emotion'] == df_valid['predicted_emotion']).mean()
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Generate detailed classification report with precision, recall, f1-score
        logger.info("\nClassification Report:")
        cr = classification_report(df_valid['emotion'], df_valid['predicted_emotion'], output_dict=True)

        # Round values for better readability
        for emotion in cr:
            if isinstance(cr[emotion], dict):
                for metric in cr[emotion]:
                    if isinstance(cr[emotion][metric], float):
                        cr[emotion][metric] = round(cr[emotion][metric], 2)

        cr_df = pd.DataFrame(cr).transpose()
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(cr_df)

        # Save classification report to file
        cr_df.to_csv(OUTPUT_FILES['report'])
        files.download(OUTPUT_FILES['report'])

        # Generate confusion matrix to visualize prediction patterns
        cm = confusion_matrix(
            df_valid['emotion'],
            df_valid['predicted_emotion'],
            labels=VALID_EMOTIONS
        )
        cm_df = pd.DataFrame(cm, index=VALID_EMOTIONS, columns=VALID_EMOTIONS)
        print("\nConfusion Matrix:")
        print(cm_df)

        # Save confusion matrix to file
        cm_df.to_csv(OUTPUT_FILES['confusion'])
        files.download(OUTPUT_FILES['confusion'])

        # Create visual heatmap of confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Emotion Classification Confusion Matrix")
        plt.ylabel("True Emotion")
        plt.xlabel("Predicted Emotion")
        plt.tight_layout()
        plt.savefig(OUTPUT_FILES['heatmap'], dpi=300, bbox_inches='tight')
        files.download(OUTPUT_FILES['heatmap'])

        # Analyze commonly confused emotion pairs
        mistakes = df_valid[df_valid['emotion'] != df_valid['predicted_emotion']]
        if len(mistakes) > 0:
            print("\nCommonly confused pairs:")
            confusion_pairs = mistakes.groupby(['emotion', 'predicted_emotion']).size().reset_index(name='count')
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
            print(confusion_pairs.head(10))

            confusion_pairs.to_csv(OUTPUT_FILES['confusion_pairs'], index=False)
            files.download(OUTPUT_FILES['confusion_pairs'])

        # Calculate per-emotion accuracy for more detailed analysis
        per_emotion_accuracy = []
        print("\nPer-emotion accuracy:")
        for emotion in VALID_EMOTIONS:
            emotion_subset = df_valid[df_valid['emotion'] == emotion]
            if len(emotion_subset) > 0:
                emotion_acc = (emotion_subset['predicted_emotion'] == emotion).mean()
                per_emotion_accuracy.append({
                    'emotion': emotion,
                    'accuracy': emotion_acc,
                    'sample_count': len(emotion_subset)
                })
                print(f"{emotion}: {emotion_acc:.4f} (n={len(emotion_subset)})")

        per_emotion_df = pd.DataFrame(per_emotion_accuracy)
        per_emotion_df.to_csv(OUTPUT_FILES['per_emotion'], index=False)
        files.download(OUTPUT_FILES['per_emotion'])

        # Visualize per-emotion accuracy with a bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x='emotion', y='accuracy', hue='emotion', data=per_emotion_df,
            palette='viridis', legend=False)
        plt.title('Accuracy by Emotion Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Emotion')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Claude_emotion_accuracy_chart.png', dpi=300, bbox_inches='tight')
        files.download('Claude_emotion_accuracy_chart.png')

        print("\nInitiating downloads for evaluation results...")
        print(f"Download links created for evaluation files")

        return accuracy, cm_df

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    """Main function to run the emotion detection pipeline."""
    # Step 1: Test API connection before proceeding
    if not test_api_connection():
        logger.error("Failed to connect to Anthropic Claude API. Please check your API key and internet connection.")
        return

    # Step 2: Get the dataset from user upload
    print("Please upload emotion_balanced.csv file:")
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # Step 3: Process the dataset in batches to handle large files efficiently
    results_df = process_dataset_batched(
        file_content,
        OUTPUT_FILES['results'],
        batch_size=20,
        max_samples=None  # Process all samples in the dataset
    )

    # Step 4: Download the main results file
    print("\nInitiating download for main results file...")
    files.download(OUTPUT_FILES['results'])
    print(f"Download link created for {OUTPUT_FILES['results']}")

    # Step 5: Evaluate model performance and generate visualizations
    if results_df is not None and not results_df.empty:
        accuracy, confusion = evaluate_results(results_df)

        print("\nProcessing complete! All results have been saved and download links have been created.")
        print("Please check your downloads folder for the following files:")
        for file_type, file_name in OUTPUT_FILES.items():
            if 'temp' not in file_type:
                print(f"- {file_name}")
        print("- Claude_emotion_accuracy_chart.png")
    else:
        print("No results to evaluate. Please check for errors above.")

if __name__ == "__main__":
    main()

# TODO: Future improvements:
# 1. Adapt code to work in non-Colab environments while maintaining simplicity
# 2. Add more customization options for visualizations
