"""
Persian Sentiment Analysis using OpenAI GPT-4o model via API

This script performs Persian sentiment analysis using the OpenAI API and automates evaluation and visualization of results.
Input: A file containing Persian texts with sentiment labels (sentiment_balanced.csv)
Output: Multiple analysis files including GPT4o_sentiment_results.csv (main results),
        confusion matrix, classification reports, and visualization charts

Purpose:
    This module provides functionality to classify sentiments in Persian text using the OpenAI API.
    It processes datasets of Persian text, detects sentiments using AI, evaluates classification
    performance, and generates analytical reports with visualizations to support sentiment analysis
    research.

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - openai (tested with version 1.12.0) 
    - tqdm (tested with version 4.66.1)
    - scikit-learn (tested with version 1.4.1)
    - matplotlib (tested with version 3.10.0)
    - seaborn (tested with version 0.13.2)
"""

# IMPORTANT: Install dependencies before running this script
# !pip install -q -U openai
# !pip install openai
# !pip install pandas numpy tqdm scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import openai
from openai import OpenAI
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
        logging.FileHandler("sentiment_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
# API key should be loaded from environment variable or config file
# NEVER hardcode credentials in source code
API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_MODEL = "gpt-4o"  # OpenAI GPT-4o model identifier
VALID_SENTIMENTS = ['negative', 'neutral', 'positive']
# Multiple encodings to try, in order of preference, for handling Persian text
ENCODING_OPTIONS = ['utf-8-sig', 'utf-8', 'windows-1256', 'cp1256', 'ISO-8859-6']
# File paths for all output files - centralized for easy modification
OUTPUT_FILES = {
    'results': "GPT4o_sentiment_results.csv",
    'temp_results': "GPT4o_sentiment_results_temp.csv",  # For atomic file operations
    'confusion': "GPT4o_sentiment_confusion_matrix.csv",
    'report': "GPT4o_sentiment_classification_report.csv",
    'heatmap': "GPT4o_sentiment_confusion_heatmap.png",
    'confusion_pairs': "GPT4o_sentiment_confusion_pairs.csv",
    'per_sentiment': "GPT4o_sentiment_accuracy.csv"
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
    Initialize the OpenAI client with the API key.
    
    Returns:
        OpenAI client object or None if initialization fails
    """
    try:
        api_key = get_api_key()
        client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI API initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI API: {str(e)}")
        return None

def test_api_connection() -> bool:
    """
    Test the connection to OpenAI API with a simple request.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    logger.info("Testing OpenAI API connection...")
    try:
        client = initialize_openai_api()
        if not client:
            return False
            
        # Send a simple test message to verify API connectivity
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "Say 'OpenAI API is working!' if you can receive this message."}
            ],
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

def batch_detect_sentiments(texts: List[str], model_name: str = DEFAULT_MODEL, max_retries: int = 3) -> Dict[str, str]:
    """
    Detect sentiments in a batch of Persian texts using OpenAI API.
    
    Args:
        texts: List of Persian text strings to analyze
        model_name: GPT4o model to use
        max_retries: Maximum number of retry attempts if API call fails
        
    Returns:
        Dictionary mapping text IDs to detected sentiments
    """
    start_time = time.time()

    # Create a prompt that instructs the model to classify sentiments
    prompt = """
You are a precise sentiment classifier for Persian text. You must classify each text into EXACTLY ONE of these three sentiments:
- negative
- neutral
- positive

IMPORTANT CONSTRAINTS:
1. Use ONLY these three sentiments - no variations, no additional sentiments
2. Return EXACTLY one sentiment per text
3. Use lowercase for all sentiment labels
4. Do not explain your choices or add any text besides the JSON object

Classify the sentiment in each of these Persian texts:

"""

    # Add each text to the prompt with a numbered identifier
    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: {text}\n\n"

    # Add the expected response format to the prompt
    prompt += """
Your response must be a valid JSON object with this exact format:
{
  "1": "sentiment_name",
  "2": "sentiment_name",
  ...
}

Where sentiment_name is ONLY one of: negative, neutral, positive.
"""

    retries = 0
    client = None

    # Implement retry logic with exponential backoff for API reliability
    while retries <= max_retries:
        try:
            # Check if we need to reinitialize the API
            if client is None:
                client = initialize_openai_api()
                if not client:
                    logger.error("Failed to initialize OpenAI API client")
                    retries += 1
                    wait_time = 2 ** retries
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue

            # Make the API request with the sentiment classification prompt
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis tool that returns only JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            # Check for valid response from API
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                logger.error("Empty or invalid response from OpenAI API")
                if retries < max_retries:
                    retries += 1
                    wait_time = 2 ** retries
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {}

            content = response.choices[0].message.content

            # Parse the JSON response from the model
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

            # Validate the sentiment results to ensure they match expected values
            validated_results = {}
            for text_id, sentiment in results.items():
                if sentiment.lower() in VALID_SENTIMENTS:
                    validated_results[text_id] = sentiment.lower()
                else:
                    logger.warning(f"Invalid sentiment '{sentiment}' detected for text {text_id}. Skipping.")

            elapsed_time = time.time() - start_time
            logger.info(f"✓ Processed {len(texts)} texts in {elapsed_time:.2f}s")
            return validated_results

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"× Error: {str(e)} (after {elapsed_time:.2f}s)")

            # Implement exponential backoff for transient errors
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
    
    This function is particularly useful for Persian text which may be encoded differently.
    
    Args:
        csv_content: CSV content as string or bytes
        
    Returns:
        DataFrame with the CSV data
        
    Raises:
        ValueError if no encoding works
    """
    # Try different encodings in order of likelihood for Persian text
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
    
    This avoids data corruption if the program is interrupted during saving.
    
    Args:
        df: DataFrame to save
        filepath: Target file path
        temp_filepath: Temporary file path (created if not provided)
        
    Returns:
        True if save successful, False otherwise
    """
    try:
        # Use atomic write pattern: write to temp file then rename
        if temp_filepath is None:
            temp_filepath = f"{filepath}.tmp"

        df.to_csv(temp_filepath, index=False, encoding='utf-8-sig')

        # Use os.replace for atomic operation
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
    Process the dataset in batches to prevent API rate limiting.
    
    Args:
        csv_content: CSV content as string or bytes
        output_path: Path to save results
        batch_size: Number of texts to process in each API call
        max_samples: Maximum number of samples to process (optional)
        
    Returns:
        DataFrame with original data and predictions added
    """
    try:
        # Load the dataset with appropriate encoding
        df = read_csv_with_multiple_encodings(csv_content)

        # Validate required columns
        if "text" not in df.columns:
            raise ValueError("No 'text' column found in CSV file. The dataset must contain a 'text' column.")

        if "sentiment" not in df.columns:
            logger.warning("No 'sentiment' column found in CSV file. Adding placeholder values.")
            df['sentiment'] = "unknown"

        # Sample the dataset if max_samples is specified
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from the dataset.")

        # Initialize predicted_sentiment column if it doesn't exist
        if 'predicted_sentiment' not in df.columns:
            df['predicted_sentiment'] = None

        # Only process rows that haven't been predicted yet
        unpredicted_df = df[df['predicted_sentiment'].isnull()].copy()
        total_to_process = len(unpredicted_df)
        processed_count = 0

        logger.info(f"Found {total_to_process} rows to process")

        overall_start = time.time()

        # Process in batches with progress tracking via tqdm
        for i in tqdm(range(0, len(unpredicted_df), batch_size)):
            batch = unpredicted_df.iloc[i:min(i+batch_size, len(unpredicted_df))]
            if batch.empty:
                continue

            # Call the API to get sentiment predictions for this batch
            sentiment_results = batch_detect_sentiments(batch['text'].tolist())

            # Update the main dataframe with predictions
            if sentiment_results:
                for idx, row in batch.iterrows():
                    batch_position = batch.index.get_loc(idx) + 1
                    text_id = str(batch_position)

                    if text_id in sentiment_results:
                        df.at[idx, 'predicted_sentiment'] = sentiment_results[text_id]
                        processed_count += 1

            # Save progress after each batch for fault tolerance
            safe_save_dataframe(df, output_path, OUTPUT_FILES['temp_results'])

            # Calculate and log progress and estimated time remaining
            if processed_count > 0:
                elapsed = time.time() - overall_start
                avg_time_per_batch = elapsed / (i + min(batch_size, len(unpredicted_df) - i)) * batch_size
                remaining_batches = (total_to_process - processed_count) / batch_size
                est_remaining = avg_time_per_batch * remaining_batches
                logger.info(f"Saved progress. Batch {i//batch_size + 1} complete. Estimated remaining time: {datetime.timedelta(seconds=int(est_remaining))}")

            # Small delay to avoid overwhelming the API
            time.sleep(1)

        # Log the final processing statistics
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
    Evaluate model performance and generate visualizations and reports.
    
    Args:
        df: DataFrame with true and predicted sentiments
        
    Returns:
        Tuple of (accuracy, confusion matrix DataFrame)
    """
    # Validate that predictions exist before evaluation
    if df is None or df.empty or df['predicted_sentiment'].isna().all():
        logger.error("No predictions were made. Evaluation cannot proceed.")
        return None, None

    # Filter to rows with valid sentiment predictions
    df_valid = df[df['predicted_sentiment'].isin(VALID_SENTIMENTS)]

    if len(df_valid) == 0:
        logger.error("No valid sentiment predictions found. Evaluation cannot proceed.")
        return None, None

    try:
        # Calculate overall accuracy
        accuracy = (df_valid['sentiment'] == df_valid['predicted_sentiment']).mean()
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Generate classification report
        logger.info("\nClassification Report:")
        cr = classification_report(df_valid['sentiment'], df_valid['predicted_sentiment'], output_dict=True)

        # Round values for better readability
        for sentiment in cr:
            if isinstance(cr[sentiment], dict):
                for metric in cr[sentiment]:
                    if isinstance(cr[sentiment][metric], float):
                        cr[sentiment][metric] = round(cr[sentiment][metric], 2)

        # Convert to DataFrame and print
        cr_df = pd.DataFrame(cr).transpose()
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(cr_df)

        # Save classification report
        cr_df.to_csv(OUTPUT_FILES['report'])
        files.download(OUTPUT_FILES['report'])

        # Generate confusion matrix
        cm = confusion_matrix(
            df_valid['sentiment'],
            df_valid['predicted_sentiment'],
            labels=VALID_SENTIMENTS
        )
        cm_df = pd.DataFrame(cm, index=VALID_SENTIMENTS, columns=VALID_SENTIMENTS)
        print("\nConfusion Matrix:")
        print(cm_df)

        # Save confusion matrix as CSV
        cm_df.to_csv(OUTPUT_FILES['confusion'])
        files.download(OUTPUT_FILES['confusion'])

        # Create confusion matrix heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Sentiment Classification Confusion Matrix")
        plt.ylabel("True Sentiment")
        plt.xlabel("Predicted Sentiment")
        plt.tight_layout()
        plt.savefig(OUTPUT_FILES['heatmap'], dpi=300, bbox_inches='tight')
        files.download(OUTPUT_FILES['heatmap'])

        # Analyze commonly confused sentiment pairs
        mistakes = df_valid[df_valid['sentiment'] != df_valid['predicted_sentiment']]
        if len(mistakes) > 0:
            print("\nCommonly confused pairs:")
            confusion_pairs = mistakes.groupby(['sentiment', 'predicted_sentiment']).size().reset_index(name='count')
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
            print(confusion_pairs.head(10))

            # Save confusion pairs analysis
            confusion_pairs.to_csv(OUTPUT_FILES['confusion_pairs'], index=False)
            files.download(OUTPUT_FILES['confusion_pairs'])

        # Calculate per-sentiment accuracy metrics
        per_sentiment_accuracy = []
        print("\nPer-sentiment accuracy:")
        for sentiment in VALID_SENTIMENTS:
            sentiment_subset = df_valid[df_valid['sentiment'] == sentiment]
            if len(sentiment_subset) > 0:
                sentiment_acc = (sentiment_subset['predicted_sentiment'] == sentiment).mean()
                per_sentiment_accuracy.append({
                    'sentiment': sentiment,
                    'accuracy': sentiment_acc,
                    'sample_count': len(sentiment_subset)
                })
                print(f"{sentiment}: {sentiment_acc:.4f} (n={len(sentiment_subset)})")

        # Save per-sentiment accuracy data
        per_sentiment_df = pd.DataFrame(per_sentiment_accuracy)
        per_sentiment_df.to_csv(OUTPUT_FILES['per_sentiment'], index=False)
        files.download(OUTPUT_FILES['per_sentiment'])

        # Create bar chart of per-sentiment accuracy
        plt.figure(figsize=(10, 6))
        sns.barplot(x='sentiment', y='accuracy', hue='sentiment', data=per_sentiment_df,
            palette='viridis', legend=False)
        plt.title('Accuracy by Sentiment Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Sentiment')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('GPT4o_sentiment_accuracy_chart.png', dpi=300, bbox_inches='tight')
        files.download('GPT4o_sentiment_accuracy_chart.png')

        print("\nInitiating downloads for evaluation results...")
        print(f"Download links created for evaluation files")

        return accuracy, cm_df

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    """
    Main execution function that orchestrates the entire sentiment analysis workflow:
    1. Test API connection
    2. Upload and process dataset
    3. Evaluate and visualize results
    """
    # Step 1: Test API connection before proceeding
    if not test_api_connection():
        logger.error("Failed to connect to OpenAI API. Please check your API key and internet connection.")
        return

    # Step 2: Upload the dataset file
    print("Please upload sentiment_balanced.csv file:")
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # Step 3: Process the dataset in batches to get sentiment predictions
    results_df = process_dataset_batched(
        file_content,
        OUTPUT_FILES['results'],
        batch_size=20,
        max_samples=None
    )

    # Step 4: Download the main results file
    print("\nInitiating download for main results file...")
    files.download(OUTPUT_FILES['results'])
    print(f"Download link created for {OUTPUT_FILES['results']}")

    # Step 5: Evaluate the results and generate visualizations if data is available
    if results_df is not None and not results_df.empty:
        accuracy, confusion = evaluate_results(results_df)

        # Step 6: Summarize all result files created
        print("\nProcessing complete! All results have been saved and download links have been created.")
        print("Please check your downloads folder for the following files:")
        for file_type, file_name in OUTPUT_FILES.items():
            if 'temp' not in file_type:
                print(f"- {file_name}")
        print("- GPT4o_sentiment_accuracy_chart.png")
    else:
        print("No results to evaluate. Please check for errors above.")

if __name__ == "__main__":
    main()

# TODO: Future improvements:
# 1. Adapt code to work in non-Colab environments while maintaining simplicity
# 2. Add more customization options for visualizations
