"""
Persian Emotion Detection using OpenAI GPT-4o model via API

This script performs Persian emotion detection using the OpenAI API and automates evaluation and visualization of results.
Input: A file containing 1800 balanced Persian texts with emotion labels (emotion_balanced.csv)
       (300 samples per emotion category)
Output: Multiple analysis files including OpenAI_emotion_results.csv (main results),
        confusion matrix, classification reports, and visualization charts

Purpose:
    This module provides functionality to classify emotions in Persian text using the OpenAI API.
    It processes datasets of Persian text, detects emotions using AI, evaluates classification
    performance, and generates analytical reports with visualizations to support emotion detection
    research.

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - openai (tested with version 1.26.0) 
    - tqdm (tested with version 4.66.1)
    - scikit-learn (tested with version 1.4.1)
    - matplotlib (tested with version 3.10.0)
    - seaborn (tested with version 0.13.2)
"""

# IMPORTANT: Install dependencies before running this script
# !pip install -q -U openai
# !pip install openai
# !pip install pandas numpy tqdm scikit-learn matplotlib seaborn

# ====== LIBRARY IMPORTS ======
# Data manipulation and analysis libraries
import pandas as pd
import numpy as np

# OpenAI API libraries
import openai
from openai import OpenAI

# Utility libraries
import time
import json
import traceback
import io
import datetime
import os
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple, Any

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Google Colab specific library for file uploads/downloads
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

# ====== CONFIGURATION CONSTANTS ======
# API key should be loaded from environment variable or config file
# NEVER hardcode credentials in source code
API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_MODEL = "gpt-4o"  # OpenAI GPT-4o model identifier
VALID_EMOTIONS = ['anger', 'fear', 'happiness', 'hate', 'sadness', 'surprise']
# Multiple encodings to try, in order of preference, for handling Persian text
ENCODING_OPTIONS = ['utf-8-sig', 'utf-8', 'windows-1256', 'cp1256', 'ISO-8859-6']
# File paths for all output files - centralized for easy modification
OUTPUT_FILES = {
    'results': "OpenAI_emotion_results.csv",
    'temp_results': "OpenAI_emotion_results_temp.csv",  # For atomic file operations
    'confusion': "OpenAI_emotion_confusion_matrix.csv",
    'report': "OpenAI_emotion_classification_report.csv",
    'heatmap': "OpenAI_emotion_confusion_heatmap.png",
    'confusion_pairs': "OpenAI_emotion_confusion_pairs.csv",
    'per_emotion': "OpenAI_emotion_accuracy.csv"
}

# Note: This implementation is specifically optimized for Google Colab to make it
# accessible for beginner researchers. The file upload/download mechanisms and fixed
# output paths are intentionally designed for the Colab environment, which provides
# a consistent and beginner-friendly interface. For production environments, these
# would be parameterized for greater flexibility.

def get_api_key() -> str:
    """
    Get the OpenAI API key from Colab secrets or manual input.

    Returns:
        str: The API key
    """
    # ======================================================================
    # HOW TO SET UP YOUR API KEY (2 SIMPLE OPTIONS)
    # ======================================================================
    # OPTION 1: RECOMMENDED - Using Colab Secrets (Secure & Persistent)
    #    - Click the 🔑 key icon in the left sidebar
    #    - Click "Add new secret"
    #    - For name: enter OPENAI_API_KEY
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
        api_key = userdata.get('OPENAI_API_KEY')
        if api_key and api_key.strip():
            logger.info("✅ API key loaded from Colab secrets")
            return api_key
    except:
        logger.info("API key not found in Colab secrets")

    logger.warning("⚠️ API key not found in Colab secrets")
    logger.warning("⚠️ For security, please add it to Colab secrets using the key icon in the sidebar")
    api_key = input("Enter your OpenAI API key (will not be stored permanently): ")
    return api_key

def initialize_openai_api():
    """
    Initialize the OpenAI API client with the provided API key.
    
    Returns:
        OpenAI client object or None if initialization fails
    """
    try:
        api_key = get_api_key()
        openai_client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI API initialized successfully")
        return openai_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI API: {str(e)}")
        return None

def test_api_connection() -> bool:
    """
    Test the OpenAI API connection by sending a simple request.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    logger.info("Testing OpenAI API connection...")
    try:
        client = initialize_openai_api()
        if not client:
            return False
            
        # Send a simple test message to verify API is working
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Say 'OpenAI API is working!' if you can receive this message."}],
            temperature=0
        )

        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            logger.info(f"API test successful! Response: {response.choices[0].message.content}")
            return True
        else:
            logger.error("API test failed. Empty or invalid response.")
            return False
    except Exception as e:
        logger.error(f"API test exception: {str(e)}")
        return False

def batch_detect_emotions(texts: List[str], model_name: str = DEFAULT_MODEL, max_retries: int = 3) -> Dict[str, str]:
    """
    Detect emotions in a batch of Persian texts using the OpenAI API.
    
    Args:
        texts: List of Persian text strings to analyze
        model_name: OpenAI model to use (default: gpt-4o)
        max_retries: Maximum number of retry attempts if API call fails
    
    Returns:
        Dict mapping text indices to detected emotions
    """
    start_time = time.time()

    # Construct a detailed prompt that instructs the model to classify emotions precisely
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

    # Add each text to be classified to the prompt
    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: {text}\n\n"

    # Add instructions for the expected JSON response format
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

    # Implement retry logic for API call resilience
    while retries <= max_retries:
        try:
            # Check if we need to reinitialize the API
            if client is None:
                client = initialize_openai_api()
                if not client:
                    raise Exception("Failed to initialize OpenAI client")

            # Make the API call with enforced JSON response format
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Use 0 temperature for most deterministic results
                response_format={"type": "json_object"}  # Force JSON response format
            )

            # Handle empty or invalid response
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                logger.error("Empty or invalid response from OpenAI API")
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}

            content = response.choices[0].message.content

            # Parse the JSON response
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

            # Validate the emotions in the response
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

            # Implement exponential backoff for retries
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
    Attempt to read a CSV file with multiple encodings to handle Persian text properly.
    
    Args:
        csv_content: CSV content as string or bytes
    
    Returns:
        pandas DataFrame containing the CSV data
    
    Raises:
        ValueError: If CSV cannot be read with any of the attempted encodings
    """
    # Try different encodings in order of preference for Persian text
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
    Safely save DataFrame to file using a temporary file to prevent data corruption.
    
    Args:
        df: pandas DataFrame to save
        filepath: Target filepath
        temp_filepath: Temporary filepath (optional)
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Use atomic file operations to prevent corrupted files
        if temp_filepath is None:
            temp_filepath = f"{filepath}.tmp"

        # Save to temp file first
        df.to_csv(temp_filepath, index=False, encoding='utf-8-sig')

        # Atomically replace the target file with the temp file
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
    Process the dataset in batches to handle large datasets efficiently.
    
    Args:
        csv_content: CSV content as string or bytes
        output_path: Path to save the results
        batch_size: Number of texts to process in each batch
        max_samples: Maximum number of samples to process (optional)
    
    Returns:
        pandas DataFrame with predicted emotions or None if processing failed
    """
    try:
        # Read the dataset with appropriate encoding
        df = read_csv_with_multiple_encodings(csv_content)

        # Validate required columns exist in the dataset
        if "text" not in df.columns:
            raise ValueError("No 'text' column found in CSV file. The dataset must contain a 'text' column.")

        if "emotion" not in df.columns:
            logger.warning("No 'emotion' column found in CSV file. Adding placeholder values.")
            df['emotion'] = "unknown"

        # Optionally limit the number of samples to process
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from the dataset.")

        # Add predicted_emotion column if it doesn't exist
        if 'predicted_emotion' not in df.columns:
            df['predicted_emotion'] = None

        # Only process rows that haven't been predicted yet (enables resuming interrupted runs)
        unpredicted_df = df[df['predicted_emotion'].isnull()].copy()
        total_to_process = len(unpredicted_df)
        processed_count = 0

        logger.info(f"Found {total_to_process} rows to process")

        overall_start = time.time()

        # Process in batches to manage API rate limits and memory use
        for i in tqdm(range(0, len(unpredicted_df), batch_size)):
            batch = unpredicted_df.iloc[i:min(i+batch_size, len(unpredicted_df))]
            if batch.empty:
                continue

            # Get emotion predictions for this batch
            emotion_results = batch_detect_emotions(batch['text'].tolist())

            # Update the dataframe with predicted emotions
            if emotion_results:
                for idx, row in batch.iterrows():
                    batch_position = batch.index.get_loc(idx) + 1
                    text_id = str(batch_position)

                    if text_id in emotion_results:
                        df.at[idx, 'predicted_emotion'] = emotion_results[text_id]
                        processed_count += 1

            # Save progress after each batch in case of interruptions
            safe_save_dataframe(df, output_path, OUTPUT_FILES['temp_results'])

            # Show progress and estimated time remaining
            if processed_count > 0:
                elapsed = time.time() - overall_start
                avg_time_per_batch = elapsed / (i + min(batch_size, len(unpredicted_df) - i)) * batch_size
                remaining_batches = (total_to_process - processed_count) / batch_size
                est_remaining = avg_time_per_batch * remaining_batches
                logger.info(f"Saved progress. Batch {i//batch_size + 1} complete. Estimated remaining time: {datetime.timedelta(seconds=int(est_remaining))}")

            # Small delay to avoid API rate limits
            time.sleep(1)

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
    Evaluate the emotion detection results and generate evaluation metrics and visualizations.
    
    Args:
        df: DataFrame containing actual and predicted emotions
    
    Returns:
        Tuple of (accuracy, confusion_matrix_dataframe) or (None, None) if evaluation failed
    """
    # Validate that predictions exist
    if df is None or df.empty or df['predicted_emotion'].isna().all():
        logger.error("No predictions were made. Evaluation cannot proceed.")
        return None, None

    # Filter to only valid emotion predictions
    df_valid = df[df['predicted_emotion'].isin(VALID_EMOTIONS)]

    if len(df_valid) == 0:
        logger.error("No valid emotion predictions found. Evaluation cannot proceed.")
        return None, None

    try:
        # Calculate overall accuracy
        accuracy = (df_valid['emotion'] == df_valid['predicted_emotion']).mean()
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Generate classification report with precision, recall, f1-score for each emotion
        logger.info("\nClassification Report:")
        cr = classification_report(df_valid['emotion'], df_valid['predicted_emotion'], output_dict=True)

        # Round metrics to 2 decimal places for readability
        for emotion in cr:
            if isinstance(cr[emotion], dict):
                for metric in cr[emotion]:
                    if isinstance(cr[emotion][metric], float):
                        cr[emotion][metric] = round(cr[emotion][metric], 2)

        # Display and save classification report
        cr_df = pd.DataFrame(cr).transpose()
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(cr_df)

        cr_df.to_csv(OUTPUT_FILES['report'])
        files.download(OUTPUT_FILES['report'])

        # Generate confusion matrix to see patterns of misclassification
        cm = confusion_matrix(
            df_valid['emotion'],
            df_valid['predicted_emotion'],
            labels=VALID_EMOTIONS
        )
        cm_df = pd.DataFrame(cm, index=VALID_EMOTIONS, columns=VALID_EMOTIONS)
        print("\nConfusion Matrix:")
        print(cm_df)

        # Save confusion matrix to CSV
        cm_df.to_csv(OUTPUT_FILES['confusion'])
        files.download(OUTPUT_FILES['confusion'])

        # Create heatmap visualization of confusion matrix
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

            # Save confused pairs analysis to CSV
            confusion_pairs.to_csv(OUTPUT_FILES['confusion_pairs'], index=False)
            files.download(OUTPUT_FILES['confusion_pairs'])

        # Calculate and display per-emotion accuracy
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

        # Save per-emotion accuracy results
        per_emotion_df = pd.DataFrame(per_emotion_accuracy)
        per_emotion_df.to_csv(OUTPUT_FILES['per_emotion'], index=False)
        files.download(OUTPUT_FILES['per_emotion'])

        # Create bar chart visualization of per-emotion accuracy
        plt.figure(figsize=(10, 6))
        sns.barplot(x='emotion', y='accuracy', hue='emotion', data=per_emotion_df,
            palette='viridis', legend=False)
        plt.title('Accuracy by Emotion Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Emotion')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('OpenAI_emotion_accuracy_chart.png', dpi=300, bbox_inches='tight')
        files.download('OpenAI_emotion_accuracy_chart.png')

        print("\nInitiating downloads for evaluation results...")
        print(f"Download links created for evaluation files")

        return accuracy, cm_df

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    """
    Main function that orchestrates the emotion detection workflow:
    1. Tests API connection
    2. Gets input file from user
    3. Processes the dataset in batches
    4. Evaluates and visualizes results
    """
    # Step 1: Verify OpenAI API connection works before proceeding
    if not test_api_connection():
        logger.error("Failed to connect to OpenAI API. Please check your API key and internet connection.")
        return

    # Step 2: Get input file from user via Colab file upload
    print("Please upload emotion_balanced.csv file:")
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # Step 3: Process the dataset in batches (main processing function)
    results_df = process_dataset_batched(
        file_content,
        OUTPUT_FILES['results'],
        batch_size=20,
        max_samples=None  # Process all samples
    )

    # Step 4: Download the main results file
    print("\nInitiating download for main results file...")
    files.download(OUTPUT_FILES['results'])
    print(f"Download link created for {OUTPUT_FILES['results']}")

    # Step 5: Evaluate results and generate visualizations
    if results_df is not None and not results_df.empty:
        accuracy, confusion = evaluate_results(results_df)

        # Step 6: Summarize the outputs for the user
        print("\nProcessing complete! All results have been saved and download links have been created.")
        print("Please check your downloads folder for the following files:")
        for file_type, file_name in OUTPUT_FILES.items():
            if 'temp' not in file_type:
                print(f"- {file_name}")
        print("- OpenAI_emotion_accuracy_chart.png")
    else:
        print("No results to evaluate. Please check for errors above.")

if __name__ == "__main__":
    main()

# TODO: Future improvements:
# 1. Adapt code to work in non-Colab environments while maintaining simplicity
# 2. Add more customization options for visualizations
