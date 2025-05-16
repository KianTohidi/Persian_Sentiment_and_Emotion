"""
Persian Emotion Detection using DeepSeek-V3 model via API call

This script performs Persian emotion detection using the DeepSeek API and automates evaluation and visualization of results.
Input: A file containing 1800 balanced Persian texts with emotion labels (emotion_balanced.csv)
       (300 samples per emotion category)
Output: Multiple analysis files including DeepSeek_emotion_results.csv (main results),
        confusion matrix, classification reports, and visualization charts

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
        logging.StreamHandler(),
        logging.FileHandler("emotion_detection.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V3 model identifier
VALID_EMOTIONS = ['anger', 'fear', 'happiness', 'hate', 'sadness', 'surprise']
# Multiple encodings to try, in order of preference, for handling Persian text
ENCODING_OPTIONS = ['utf-8-sig', 'utf-8', 'windows-1256', 'cp1256', 'ISO-8859-6']
# File paths for all output files - centralized for easy modification
OUTPUT_FILES = {
    'results': "DeepSeek_emotion_results.csv",
    'temp_results': "DeepSeek_emotion_results_temp.csv",
    'confusion': "DeepSeek_emotion_confusion_matrix.csv",
    'report': "DeepSeek_emotion_classification_report.csv",
    'heatmap': "DeepSeek_emotion_confusion_heatmap.png",
    'confusion_pairs': "DeepSeek_emotion_confusion_pairs.csv",
    'per_emotion': "DeepSeek_emotion_accuracy.csv"
}

def get_api_key() -> str:
    """
    Get the DeepSeek API key from Colab secrets or manual input.

    Returns:
        str: The API key
    """
    try:
        from google.colab import userdata
        api_key = userdata.get('DEEPSEEK_API_KEY')
        if api_key and api_key.strip():
            logger.info("✅ API key loaded from Colab secrets")
            return api_key
    except:
        logger.info("API key not found in Colab secrets")

    logger.warning("⚠️ API key not found in Colab secrets")
    logger.warning("⚠️ For security, please add it to Colab secrets using the key icon in the sidebar")
    api_key = input("Enter your DeepSeek API key (will not be stored permanently): ")
    return api_key

def test_api_connection() -> bool:
    """Test the connection with the DeepSeek API."""
    logger.info("Testing DeepSeek API connection...")
    try:
        api_key = get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Send a simple test request to verify API connectivity
        test_payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'DeepSeek API is working!' if you can receive this message."}
            ],
            "max_tokens": 20
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=test_payload)

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"API test successful! Response: {content}")
            return True
        else:
            logger.error(f"API test failed. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"API test exception: {str(e)}")
        return False

def batch_detect_emotions(texts: List[str], model: str = DEFAULT_MODEL, max_retries: int = 3) -> Dict[str, str]:
    """
    Process multiple texts in a single API call to detect emotions.
    
    Args:
        texts: List of Persian text strings to analyze
        model: DeepSeek model name to use. Defaults to DEFAULT_MODEL
        max_retries: Maximum number of retry attempts on failure
        
    Returns:
        Dict[str, str]: A dictionary where keys are text IDs and values are emotion labels
    """
    start_time = time.time()

    # Create a structured prompt that instructs the LLM to identify emotions
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

    # Add each text to the prompt with numbering for identification
    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: {text}\n\n"

    # Add format instructions for the JSON response structure
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

    # Implement retry logic for robustness against API failures
    while retries <= max_retries:
        try:
            api_key = get_api_key()

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Configure API request with forced JSON output for consistent parsing
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You analyze emotions in Persian texts and return results in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,  # Set to 0 for deterministic output
                "max_tokens": 500,
                "response_format": {"type": "json_object"}  # Force JSON response
            }

            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

            # Handle HTTP errors from the API
            if response.status_code != 200:
                error_message = f"API Error: Status code {response.status_code}"
                logger.error(error_message)
                logger.error(f"Response: {response.text}")
    
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}       

            response_data = response.json()

            # Extract the response content from DeepSeek's API structure
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "{}")

                if isinstance(content, str):
                    try:
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
                else:
                    results = content
            else:
                logger.error(f"Unexpected response format: {response_data}")
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}

            # Validate that all emotions are in our accepted list and normalize to lowercase
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
    # Try each encoding in sequence until one works
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
        if temp_filepath is None:
            temp_filepath = f"{filepath}.tmp"

        # First write to a temporary file to avoid corrupting the destination
        df.to_csv(temp_filepath, index=False, encoding='utf-8-sig')

        # Atomic file replacement to ensure data integrity
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
    Process a dataset in batches, detecting emotions using DeepSeek API.
    
    Args:
        csv_content: Raw CSV content as string or bytes
        output_path: Path to save the results
        batch_size: Number of texts to process in each batch
        max_samples: Maximum number of samples to process
        
    Returns:
        DataFrame with the processed results, or None on failure
    """
    try:
        # Load the dataset with support for different encodings
        df = read_csv_with_multiple_encodings(csv_content)

        # Validate required columns exist in the dataset
        if "text" not in df.columns:
            raise ValueError("No 'text' column found in CSV file. The dataset must contain a 'text' column.")

        if "emotion" not in df.columns:
            logger.warning("No 'emotion' column found in CSV file. Adding placeholder values.")
            df['emotion'] = "unknown"

        # Apply sampling if requested (for testing or limiting API calls)
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from the dataset.")

        # Add prediction column if it doesn't exist
        if 'predicted_emotion' not in df.columns:
            df['predicted_emotion'] = None

        # Filter to only process rows that don't have predictions yet
        unpredicted_df = df[df['predicted_emotion'].isnull()].copy()
        total_to_process = len(unpredicted_df)
        processed_count = 0

        logger.info(f"Found {total_to_process} rows to process")

        overall_start = time.time()

        # Process in batches to optimize API calls and enable progress saving
        for i in tqdm(range(0, len(unpredicted_df), batch_size)):
            batch = unpredicted_df.iloc[i:min(i+batch_size, len(unpredicted_df))]
            if batch.empty:
                continue

            # Call the emotion detection API for this batch
            emotion_results = batch_detect_emotions(batch['text'].tolist())

            if emotion_results:
                for idx, row in batch.iterrows():
                    batch_position = batch.index.get_loc(idx) + 1
                    text_id = str(batch_position)

                    if text_id in emotion_results:
                        df.at[idx, 'predicted_emotion'] = emotion_results[text_id]
                        processed_count += 1

            # Save progress after each batch for resumability
            safe_save_dataframe(df, output_path, OUTPUT_FILES['temp_results'])

            if processed_count > 0:
                # Display estimated time remaining based on current progress
                elapsed = time.time() - overall_start
                avg_time_per_batch = elapsed / (i + min(batch_size, len(unpredicted_df) - i)) * batch_size
                remaining_batches = (total_to_process - processed_count) / batch_size
                est_remaining = avg_time_per_batch * remaining_batches
                logger.info(f"Saved progress. Batch {i//batch_size + 1} complete. Estimated remaining time: {datetime.timedelta(seconds=int(est_remaining))}")

            time.sleep(1)  # Brief pause to avoid API rate limits

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
    if df is None or df.empty or df['predicted_emotion'].isna().all():
        logger.error("No predictions were made. Evaluation cannot proceed.")
        return None, None

    # Filter to only include valid emotion predictions
    df_valid = df[df['predicted_emotion'].isin(VALID_EMOTIONS)]

    if len(df_valid) == 0:
        logger.error("No valid emotion predictions found. Evaluation cannot proceed.")
        return None, None

    try:
        # Calculate and report overall accuracy
        accuracy = (df_valid['emotion'] == df_valid['predicted_emotion']).mean()
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Generate detailed classification report with precision, recall, and F1-score
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

        # Save classification report to file and enable download
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

        # Save confusion matrix to file and enable download
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

        # Analyze common misclassifications to identify patterns
        mistakes = df_valid[df_valid['emotion'] != df_valid['predicted_emotion']]
        if len(mistakes) > 0:
            print("\nCommonly confused pairs:")
            confusion_pairs = mistakes.groupby(['emotion', 'predicted_emotion']).size().reset_index(name='count')
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
            print(confusion_pairs.head(10))

            confusion_pairs.to_csv(OUTPUT_FILES['confusion_pairs'], index=False)
            files.download(OUTPUT_FILES['confusion_pairs'])

        # Calculate and visualize per-emotion accuracy
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

        # Create bar chart of per-emotion accuracy
        plt.figure(figsize=(10, 6))
        sns.barplot(x='emotion', y='accuracy', hue='emotion', data=per_emotion_df,
            palette='viridis', legend=False)
        plt.title('Accuracy by Emotion Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Emotion')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('DeepSeek_emotion_accuracy_chart.png', dpi=300, bbox_inches='tight')
        files.download('DeepSeek_emotion_accuracy_chart.png')

        print("\nInitiating downloads for evaluation results...")
        print(f"Download links created for evaluation files")

        return accuracy, cm_df

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    """Main function to run the emotion detection pipeline."""
    # First validate API connectivity before proceeding
    if not test_api_connection():
        logger.error("Failed to connect to DeepSeek API. Please check your API key and internet connection.")
        return

    # Request user to upload the dataset file through Colab interface
    print("Please upload emotion_balanced.csv file:")
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # Process the dataset in batches with progress tracking and saving
    results_df = process_dataset_batched(
        file_content,
        OUTPUT_FILES['results'],
        batch_size=20,
        max_samples=None
    )

    # Enable download of the complete results file
    print("\nInitiating download for main results file...")
    files.download(OUTPUT_FILES['results'])
    print(f"Download link created for {OUTPUT_FILES['results']}")

    # Run evaluation metrics and visualizations if processing was successful
    if results_df is not None and not results_df.empty:
        accuracy, confusion = evaluate_results(results_df)

        # Summarize the outputs for the user
        print("\nProcessing complete! All results have been saved and download links have been created.")
        print("Please check your downloads folder for the following files:")
        for file_type, file_name in OUTPUT_FILES.items():
            if 'temp' not in file_type:
                print(f"- {file_name}")
        print("- DeepSeek_emotion_accuracy_chart.png")
    else:
        print("No results to evaluate. Please check for errors above.")

if __name__ == "__main__":
    main()

# TODO: Future improvements:
# 1. Adapt code to work in non-Colab environments while maintaining simplicity
# 2. Add more customization options for visualizations
