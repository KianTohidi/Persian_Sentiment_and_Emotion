"""
Persian Emotion Detection using Google Gemini Flash 2.0 model via API

This script performs Persian emotion detection using the Google Gemini API and automates evaluation and visualization of results.
Input: A file containing 1800 balanced Persian texts with emotion labels (emotion_balanced.csv)
       (300 samples per emotion category)
Output: Multiple analysis files including Gemini_emotion_results.csv (main results),
        confusion matrix, classification reports, and visualization charts

Purpose:
    This module provides functionality to classify emotions in Persian text using the Google Gemini API.
    It processes datasets of Persian text, detects emotions using AI, evaluates classification
    performance, and generates analytical reports with visualizations to support emotion detection
    research.

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - google-genai (tested with version 1.14.0) 
    - tqdm (tested with version 4.66.1)
    - scikit-learn (tested with version 1.4.1)
    - matplotlib (tested with version 3.10.0)
    - seaborn (tested with version 0.13.2)
"""

# IMPORTANT: Install dependencies before running this script
# !pip install -q -U google-genai
# !pip install google-generativeai
# !pip install pandas numpy tqdm scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import google.generativeai as genai
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
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
DEFAULT_MODEL = "gemini-2.0-flash"  # Gemini 2.0 Flash model identifier
VALID_EMOTIONS = ['anger', 'fear', 'happiness', 'hate', 'sadness', 'surprise']
# Multiple encodings to try, in order of preference, for handling Persian text
ENCODING_OPTIONS = ['utf-8-sig', 'utf-8', 'windows-1256', 'cp1256', 'ISO-8859-6']
# File paths for all output files - centralized for easy modification
OUTPUT_FILES = {
    'results': "Gemini_emotion_results.csv",
    'temp_results': "Gemini_emotion_results_temp.csv",  # For atomic file operations
    'confusion': "Gemini_emotion_confusion_matrix.csv",
    'report': "Gemini_emotion_classification_report.csv",
    'heatmap': "Gemini_emotion_confusion_heatmap.png",
    'confusion_pairs': "Gemini_emotion_confusion_pairs.csv",
    'per_emotion': "Gemini_emotion_accuracy.csv"
}

# Note: This implementation is specifically optimized for Google Colab to make it
# accessible for beginner researchers. The file upload/download mechanisms and fixed
# output paths are intentionally designed for the Colab environment, which provides
# a consistent and beginner-friendly interface. For production environments, these
# would be parameterized for greater flexibility.

def get_api_key() -> str:
    """
    Get the Google Gemini API key from Colab secrets or manual input.

    Returns:
        str: The API key
    """
    # ======================================================================
    # HOW TO SET UP YOUR API KEY (2 SIMPLE OPTIONS)
    # ======================================================================
    # OPTION 1: RECOMMENDED - Using Colab Secrets (Secure & Persistent)
    #    - Click the 🔑 key icon in the left sidebar
    #    - Click "Add new secret"
    #    - For name: enter GOOGLE_API_KEY
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
        api_key = userdata.get('GOOGLE_API_KEY')
        if api_key and api_key.strip():
            logger.info("✅ API key loaded from Colab secrets")
            return api_key
    except:
        logger.info("API key not found in Colab secrets")

    logger.warning("⚠️ API key not found in Colab secrets")
    logger.warning("⚠️ For security, please add it to Colab secrets using the key icon in the sidebar")
    api_key = input("Enter your Google Gemini API key (will not be stored permanently): ")
    return api_key

def initialize_gemini_api():
    """
    Initialize the Google Gemini API with the provided API key.
    Returns success/failure status as boolean.
    """
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        logger.info("✅ Gemini API initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {str(e)}")
        return False

def test_api_connection() -> bool:
    """
    Tests connection to the Google Gemini API by sending a simple prompt.
    Essential step to verify API access before processing the dataset.
    """
    logger.info("Testing Google Gemini API connection...")
    try:
        if not initialize_gemini_api():
            return False
            
        model = genai.GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content("Say 'Google Gemini API is working!' if you can receive this message.")

        if response and hasattr(response, 'text'):
            logger.info(f"API test successful! Response: {response.text}")
            return True
        else:
            logger.error("API test failed. Empty or invalid response.")
            return False
    except Exception as e:
        logger.error(f"API test exception: {str(e)}")
        return False

def batch_detect_emotions(texts: List[str], model_name: str = DEFAULT_MODEL, max_retries: int = 3) -> Dict[str, str]:
    """
    Core emotion detection function that sends Persian texts to Gemini API.
    Uses a carefully crafted prompt to ensure consistent emotion classification.
    Implements retry logic for API resilience.
    
    Args:
        texts: List of Persian text samples to classify
        model_name: Gemini model to use (default: gemini-2.0-flash)
        max_retries: Number of retry attempts for API failures
        
    Returns:
        Dictionary mapping text IDs to detected emotions
    """
    start_time = time.time()

    # Construct the prompt with clear instructions for the Gemini model
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

    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: {text}\n\n"

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

    # Implement retry loop with exponential backoff
    while retries <= max_retries:
        try:
            # Check if we need to reinitialize the API
            try:
                # Simple check if genai is configured
                _ = genai.get_default_api_key()
            except:
                initialize_gemini_api()

            # Configure the model with zero temperature for deterministic outputs
            generation_config = {
                "temperature": 0,
                "max_output_tokens": 500,
                "response_mime_type": "application/json"
            }

            model = genai.GenerativeModel(model_name, generation_config=generation_config)

            response = model.generate_content(
                [
                    {"role": "user", "parts": [prompt]}
                ]
            )

            if not response or not hasattr(response, 'text'):
                logger.error("Empty or invalid response from Gemini API")
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}

            content = response.text

            # Parse JSON response and validate emotion labels
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

            # Validate that each emotion is in our predefined list
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
    Attempts to read CSV with various encodings to handle Persian text correctly.
    Persian text can have encoding issues, so this tries multiple encodings.
    """
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
    Safely saves DataFrame to CSV using atomic write operations.
    This prevents data corruption if the process is interrupted during saving.
    """
    try:
        if temp_filepath is None:
            temp_filepath = f"{filepath}.tmp"

        df.to_csv(temp_filepath, index=False, encoding='utf-8-sig')

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
    Main processing pipeline that:
    1. Reads the dataset
    2. Processes texts in small batches to avoid API rate limits
    3. Saves progress incrementally to prevent data loss
    4. Provides progress updates and estimated completion time
    
    Args:
        csv_content: CSV file content (either path or bytes)
        output_path: Where to save results
        batch_size: How many samples to process in each API call
        max_samples: Optional limit on dataset size
        
    Returns:
        DataFrame with original texts and predicted emotions
    """
    try:
        # Step 1: Read and validate the dataset
        df = read_csv_with_multiple_encodings(csv_content)

        if "text" not in df.columns:
            raise ValueError("No 'text' column found in CSV file. The dataset must contain a 'text' column.")

        if "emotion" not in df.columns:
            logger.warning("No 'emotion' column found in CSV file. Adding placeholder values.")
            df['emotion'] = "unknown"

        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from the dataset.")

        # Add predicted_emotion column if it doesn't exist
        if 'predicted_emotion' not in df.columns:
            df['predicted_emotion'] = None

        # Step 2: Filter to get only rows that haven't been processed yet
        unpredicted_df = df[df['predicted_emotion'].isnull()].copy()
        total_to_process = len(unpredicted_df)
        processed_count = 0

        logger.info(f"Found {total_to_process} rows to process")

        overall_start = time.time()

        # Step 3: Process in batches with progress tracking
        for i in tqdm(range(0, len(unpredicted_df), batch_size)):
            batch = unpredicted_df.iloc[i:min(i+batch_size, len(unpredicted_df))]
            if batch.empty:
                continue

            # Send batch to Gemini API for emotion detection
            emotion_results = batch_detect_emotions(batch['text'].tolist())

            if emotion_results:
                for idx, row in batch.iterrows():
                    batch_position = batch.index.get_loc(idx) + 1
                    text_id = str(batch_position)

                    if text_id in emotion_results:
                        df.at[idx, 'predicted_emotion'] = emotion_results[text_id]
                        processed_count += 1

            # Step 4: Save progress after each batch (checkpoint)
            safe_save_dataframe(df, output_path, OUTPUT_FILES['temp_results'])

            # Calculate and display progress statistics
            if processed_count > 0:
                elapsed = time.time() - overall_start
                avg_time_per_batch = elapsed / (i + min(batch_size, len(unpredicted_df) - i)) * batch_size
                remaining_batches = (total_to_process - processed_count) / batch_size
                est_remaining = avg_time_per_batch * remaining_batches
                logger.info(f"Saved progress. Batch {i//batch_size + 1} complete. Estimated remaining time: {datetime.timedelta(seconds=int(est_remaining))}")

            time.sleep(1)  # Small delay to avoid overloading the API

        # Step 5: Summarize processing statistics
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
    Comprehensive evaluation function that:
    1. Calculates accuracy metrics
    2. Generates classification reports
    3. Creates confusion matrix
    4. Produces visualizations
    5. Analyzes error patterns
    
    Args:
        df: DataFrame with true and predicted emotion labels
        
    Returns:
        Tuple of (accuracy, confusion_matrix_df)
    """
    # Validate input data before evaluation
    if df is None or df.empty or df['predicted_emotion'].isna().all():
        logger.error("No predictions were made. Evaluation cannot proceed.")
        return None, None

    df_valid = df[df['predicted_emotion'].isin(VALID_EMOTIONS)]

    if len(df_valid) == 0:
        logger.error("No valid emotion predictions found. Evaluation cannot proceed.")
        return None, None

    try:
        # Step 1: Calculate overall accuracy
        accuracy = (df_valid['emotion'] == df_valid['predicted_emotion']).mean()
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Step 2: Generate detailed classification report
        logger.info("\nClassification Report:")
        cr = classification_report(df_valid['emotion'], df_valid['predicted_emotion'], output_dict=True)

        # Round metrics to 2 decimal places for readability
        for emotion in cr:
            if isinstance(cr[emotion], dict):
                for metric in cr[emotion]:
                    if isinstance(cr[emotion][metric], float):
                        cr[emotion][metric] = round(cr[emotion][metric], 2)

        cr_df = pd.DataFrame(cr).transpose()
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(cr_df)

        # Save classification report
        cr_df.to_csv(OUTPUT_FILES['report'])
        files.download(OUTPUT_FILES['report'])

        # Step 3: Generate confusion matrix
        cm = confusion_matrix(
            df_valid['emotion'],
            df_valid['predicted_emotion'],
            labels=VALID_EMOTIONS
        )
        cm_df = pd.DataFrame(cm, index=VALID_EMOTIONS, columns=VALID_EMOTIONS)
        print("\nConfusion Matrix:")
        print(cm_df)

        # Save confusion matrix
        cm_df.to_csv(OUTPUT_FILES['confusion'])
        files.download(OUTPUT_FILES['confusion'])

        # Step 4: Create heatmap visualization of confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Emotion Classification Confusion Matrix")
        plt.ylabel("True Emotion")
        plt.xlabel("Predicted Emotion")
        plt.tight_layout()
        plt.savefig(OUTPUT_FILES['heatmap'], dpi=300, bbox_inches='tight')
        files.download(OUTPUT_FILES['heatmap'])

        # Step 5: Analyze common misclassification patterns
        mistakes = df_valid[df_valid['emotion'] != df_valid['predicted_emotion']]
        if len(mistakes) > 0:
            print("\nCommonly confused pairs:")
            confusion_pairs = mistakes.groupby(['emotion', 'predicted_emotion']).size().reset_index(name='count')
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
            print(confusion_pairs.head(10))

            confusion_pairs.to_csv(OUTPUT_FILES['confusion_pairs'], index=False)
            files.download(OUTPUT_FILES['confusion_pairs'])

        # Step 6: Calculate and visualize per-emotion accuracy
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
        plt.savefig('Gemini_emotion_accuracy_chart.png', dpi=300, bbox_inches='tight')
        files.download('Gemini_emotion_accuracy_chart.png')

        print("\nInitiating downloads for evaluation results...")
        print(f"Download links created for evaluation files")

        return accuracy, cm_df

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    """
    Main function that orchestrates the entire pipeline:
    1. Tests API connection
    2. Handles file upload
    3. Processes the dataset
    4. Evaluates and visualizes results
    """
    # Step 1: Verify API connection before starting
    if not test_api_connection():
        logger.error("Failed to connect to Google Gemini API. Please check your API key and internet connection.")
        return

    # Step 2: Upload the dataset file
    print("Please upload emotion_balanced.csv file:")
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # Step 3: Process the dataset and get results
    results_df = process_dataset_batched(
        file_content,
        OUTPUT_FILES['results'],
        batch_size=20,
        max_samples=None
    )

    # Download the complete results file
    print("\nInitiating download for main results file...")
    files.download(OUTPUT_FILES['results'])
    print(f"Download link created for {OUTPUT_FILES['results']}")

    # Step 4: Evaluate and visualize if we have results
    if results_df is not None and not results_df.empty:
        accuracy, confusion = evaluate_results(results_df)

        # Step 5: Summarize the process completion
        print("\nProcessing complete! All results have been saved and download links have been created.")
        print("Please check your downloads folder for the following files:")
        for file_type, file_name in OUTPUT_FILES.items():
            if 'temp' not in file_type:
                print(f"- {file_name}")
        print("- Gemini_emotion_accuracy_chart.png")
    else:
        print("No results to evaluate. Please check for errors above.")

# Entry point of the script
if __name__ == "__main__":
    main()

# TODO: Future improvements:
# 1. Adapt code to work in non-Colab environments while maintaining simplicity
# 2. Add more customization options for visualizations
