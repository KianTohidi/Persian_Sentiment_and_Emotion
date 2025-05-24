"""
Persian Sentiment Analysis using Anthropic Claude 3.7 Sonnet model via API

This script performs Persian sentiment analysis using the Anthropic Claude API and automates evaluation and visualization of results.
Input: A file containing Persian texts with sentiment labels (sentiment_balanced.csv)
Output: Multiple analysis files including Claude_sentiment_results.csv (main results),
        confusion matrix, classification reports, and visualization charts

Purpose:
    This module provides functionality to classify sentiments in Persian text using the Anthropic Claude API.
    It processes datasets of Persian text, detects sentiments using AI, evaluates classification
    performance, and generates analytical reports with visualizations to support sentiment analysis
    research.

Dependencies:
    - pandas (tested with version 2.2.2)
    - numpy (tested with version 2.0.2)
    - anthropic (tested with version 0.21.1) 
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

# ========== LOGGING SETUP ==========
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

# ========== CONFIGURATION ==========
# Constants
# API key should be loaded from environment variable or config file
# NEVER hardcode credentials in source code
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"  # Claude 3.7 Sonnet model identifier
VALID_SENTIMENTS = ['negative', 'neutral', 'positive']
# Multiple encodings to try, in order of preference, for handling Persian text
ENCODING_OPTIONS = ['utf-8-sig', 'utf-8', 'windows-1256', 'cp1256', 'ISO-8859-6']
# File paths for all output files - centralized for easy modification
OUTPUT_FILES = {
    'results': "Claude_sentiment_results.csv",
    'temp_results': "Claude_sentiment_results_temp.csv",  # For atomic file operations
    'confusion': "Claude_sentiment_confusion_matrix.csv",
    'report': "Claude_sentiment_classification_report.csv",
    'heatmap': "Claude_sentiment_confusion_heatmap.png",
    'confusion_pairs': "Claude_sentiment_confusion_pairs.csv",
    'per_sentiment': "Claude_sentiment_accuracy.csv"
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

def initialize_claude_api():
    # Initialize the Anthropic client with API key for Claude access
    try:
        api_key = get_api_key()
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("✅ Claude API initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Claude API: {str(e)}")
        return None

def test_api_connection() -> bool:
    # Test the connection to Claude API with a simple request
    logger.info("Testing Anthropic Claude API connection...")
    try:
        client = initialize_claude_api()
        if not client:
            return False
            
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say 'Anthropic Claude API is working!' if you can receive this message."}
            ]
        )

        if response and hasattr(response, 'content'):
            response_text = response.content[0].text
            logger.info(f"API test successful! Response: {response_text}")
            return True
        else:
            logger.error("API test failed. Empty or invalid response.")
            return False
    except Exception as e:
        logger.error(f"API test exception: {str(e)}")
        return False

def batch_detect_sentiments(texts: List[str], model_name: str = DEFAULT_MODEL, max_retries: int = 3) -> Dict[str, str]:
    # ↓↓↓ Main sentiment detection function that sends texts to Claude API in batches ↓↓↓
    start_time = time.time()
    client = None

    # ↓↓↓ Construct a detailed prompt for Claude to ensure accurate sentiment classification ↓↓↓
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

    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: {text}\n\n"

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

    # ↓↓↓ Implement exponential backoff retry mechanism for API resilience ↓↓↓
    while retries <= max_retries:
        try:
            # Check if we need to reinitialize the API
            if client is None:
                client = initialize_claude_api()
                if client is None:
                    raise ValueError("Failed to initialize Claude API")

            # ↓↓↓ Send request to Claude API with temperature=0 for consistent results ↓↓↓
            response = client.messages.create(
                model=model_name,
                max_tokens=500,
                temperature=0,  # Use deterministic output for consistency
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a sentiment analysis tool that only outputs valid JSON. Never explain or add extra text."
            )

            if not response or not hasattr(response, 'content') or not response.content:
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
                # ↓↓↓ Extract JSON from Claude's response using regex in case it includes explanatory text ↓↓↓
                import re
                json_match = re.search(r'(\{[\s\S]*\})', content)
                if json_match:
                    content = json_match.group(1)
                
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

            # ↓↓↓ Validate that all sentiments conform to our accepted values (negative, neutral, positive) ↓↓↓
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
    # ↓↓↓ Try multiple encodings to handle Persian text properly - Persian text may use different encodings ↓↓↓
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
    # ↓↓↓ Use atomic file operations to prevent data corruption during saves ↓↓↓
    try:
        if temp_filepath is None:
            temp_filepath = f"{filepath}.tmp"

        # First write to a temporary file
        df.to_csv(temp_filepath, index=False, encoding='utf-8-sig')

        # Then atomically replace the target file with the temp file
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
    try:
        # ↓↓↓ Read the dataset with appropriate encoding for Persian text ↓↓↓
        df = read_csv_with_multiple_encodings(csv_content)

        if "text" not in df.columns:
            raise ValueError("No 'text' column found in CSV file. The dataset must contain a 'text' column.")

        if "sentiment" not in df.columns:
            logger.warning("No 'sentiment' column found in CSV file. Adding placeholder values.")
            df['sentiment'] = "unknown"

        # ↓↓↓ Optionally limit dataset size for faster processing during development ↓↓↓
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from the dataset.")

        if 'predicted_sentiment' not in df.columns:
            df['predicted_sentiment'] = None

        # ↓↓↓ Filter to only process texts that haven't been analyzed yet (enables resuming interrupted runs) ↓↓↓
        unpredicted_df = df[df['predicted_sentiment'].isnull()].copy()
        total_to_process = len(unpredicted_df)
        processed_count = 0

        logger.info(f"Found {total_to_process} rows to process")

        overall_start = time.time()

        # ↓↓↓ Process in batches for efficiency and to avoid API rate limits ↓↓↓
        for i in tqdm(range(0, len(unpredicted_df), batch_size)):
            batch = unpredicted_df.iloc[i:min(i+batch_size, len(unpredicted_df))]
            if batch.empty:
                continue

            # Get sentiment predictions for current batch
            sentiment_results = batch_detect_sentiments(batch['text'].tolist())

            if sentiment_results:
                for idx, row in batch.iterrows():
                    batch_position = batch.index.get_loc(idx) + 1
                    text_id = str(batch_position)

                    if text_id in sentiment_results:
                        df.at[idx, 'predicted_sentiment'] = sentiment_results[text_id]
                        processed_count += 1

            # ↓↓↓ Save progress after each batch for resilience against crashes or interruptions ↓↓↓
            safe_save_dataframe(df, output_path, OUTPUT_FILES['temp_results'])

            if processed_count > 0:
                # ↓↓↓ Calculate and display estimated remaining time based on current progress ↓↓↓
                elapsed = time.time() - overall_start
                avg_time_per_batch = elapsed / (i + min(batch_size, len(unpredicted_df) - i)) * batch_size
                remaining_batches = (total_to_process - processed_count) / batch_size
                est_remaining = avg_time_per_batch * remaining_batches
                logger.info(f"Saved progress. Batch {i//batch_size + 1} complete. Estimated remaining time: {datetime.timedelta(seconds=int(est_remaining))}")

            time.sleep(1)  # Prevent API rate limiting

        # ↓↓↓ Log completion statistics for the entire processing job ↓↓↓
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
    # ↓↓↓ Generate comprehensive evaluation metrics and visualizations for sentiment analysis results ↓↓↓
    if df is None or df.empty or df['predicted_sentiment'].isna().all():
        logger.error("No predictions were made. Evaluation cannot proceed.")
        return None, None

    # ↓↓↓ Filter to only include predictions with valid sentiment values ↓↓↓
    df_valid = df[df['predicted_sentiment'].isin(VALID_SENTIMENTS)]

    if len(df_valid) == 0:
        logger.error("No valid sentiment predictions found. Evaluation cannot proceed.")
        return None, None

    try:
        # ↓↓↓ Calculate overall accuracy - the proportion of correct predictions ↓↓↓
        accuracy = (df_valid['sentiment'] == df_valid['predicted_sentiment']).mean()
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        print(f"Overall accuracy: {accuracy:.4f}")

        # ↓↓↓ Generate detailed classification report (precision, recall, f1-score) ↓↓↓
        logger.info("\nClassification Report:")
        cr = classification_report(df_valid['sentiment'], df_valid['predicted_sentiment'], output_dict=True)

        for sentiment in cr:
            if isinstance(cr[sentiment], dict):
                for metric in cr[sentiment]:
                    if isinstance(cr[sentiment][metric], float):
                        cr[sentiment][metric] = round(cr[sentiment][metric], 2)

        cr_df = pd.DataFrame(cr).transpose()
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(cr_df)


        cr_df.to_csv(OUTPUT_FILES['report'])
        files.download(OUTPUT_FILES['report'])

        # ↓↓↓ Create confusion matrix to visualize prediction patterns for each sentiment class ↓↓↓
        cm = confusion_matrix(
            df_valid['sentiment'],
            df_valid['predicted_sentiment'],
            labels=VALID_SENTIMENTS
        )
        cm_df = pd.DataFrame(cm, index=VALID_SENTIMENTS, columns=VALID_SENTIMENTS)
        print("\nConfusion Matrix:")
        print(cm_df)

        cm_df.to_csv(OUTPUT_FILES['confusion'])
        files.download(OUTPUT_FILES['confusion'])

        # ↓↓↓ Create heatmap visualization of confusion matrix for easier interpretation ↓↓↓
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Sentiment Classification Confusion Matrix")
        plt.ylabel("True Sentiment")
        plt.xlabel("Predicted Sentiment")
        plt.tight_layout()
        plt.savefig(OUTPUT_FILES['heatmap'], dpi=300, bbox_inches='tight')
        files.download(OUTPUT_FILES['heatmap'])

        # ↓↓↓ Analyze commonly confused sentiment pairs to identify classification challenges ↓↓↓
        mistakes = df_valid[df_valid['sentiment'] != df_valid['predicted_sentiment']]
        if len(mistakes) > 0:
            print("\nCommonly confused pairs:")
            confusion_pairs = mistakes.groupby(['sentiment', 'predicted_sentiment']).size().reset_index(name='count')
            confusion_pairs = confusion_pairs.sort_values('count', ascending=False)
            print(confusion_pairs.head(10))

            confusion_pairs.to_csv(OUTPUT_FILES['confusion_pairs'], index=False)
            files.download(OUTPUT_FILES['confusion_pairs'])

        # ↓↓↓ Calculate per-sentiment accuracy to identify which sentiments are easier/harder to classify ↓↓↓
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

        per_sentiment_df = pd.DataFrame(per_sentiment_accuracy)
        per_sentiment_df.to_csv(OUTPUT_FILES['per_sentiment'], index=False)
        files.download(OUTPUT_FILES['per_sentiment'])

        # ↓↓↓ Create bar chart for per-sentiment accuracy for visual comparison ↓↓↓
        plt.figure(figsize=(10, 6))
        sns.barplot(x='sentiment', y='accuracy', hue='sentiment', data=per_sentiment_df,
            palette='viridis', legend=False)
        plt.title('Accuracy by Sentiment Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Sentiment')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Claude_sentiment_accuracy_chart.png', dpi=300, bbox_inches='tight')
        files.download('Claude_sentiment_accuracy_chart.png')

        print("\nInitiating downloads for evaluation results...")
        print(f"Download links created for evaluation files")

        return accuracy, cm_df

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    # ↓↓↓ Main execution flow: Test API connection, process dataset, evaluate results ↓↓↓
    if not test_api_connection():
        logger.error("Failed to connect to Anthropic Claude API. Please check your API key and internet connection.")
        return

    # ↓↓↓ Prompt user to upload the Persian sentiment dataset file ↓↓↓
    print("Please upload sentiment_balanced.csv file:")
    uploaded = files.upload()

    filename = list(uploaded.keys())[0]
    file_content = uploaded[filename]

    # ↓↓↓ Process the dataset in batches to manage memory and API rate limits ↓↓↓
    results_df = process_dataset_batched(
        file_content,
        OUTPUT_FILES['results'],
        batch_size=20,
        max_samples=None
    )

    print("\nInitiating download for main results file...")
    files.download(OUTPUT_FILES['results'])
    print(f"Download link created for {OUTPUT_FILES['results']}")

    # ↓↓↓ If processing was successful, evaluate the model's performance ↓↓↓
    if results_df is not None and not results_df.empty:
        accuracy, confusion = evaluate_results(results_df)

        print("\nProcessing complete! All results have been saved and download links have been created.")
        print("Please check your downloads folder for the following files:")
        for file_type, file_name in OUTPUT_FILES.items():
            if 'temp' not in file_type:
                print(f"- {file_name}")
        print("- Claude_sentiment_accuracy_chart.png")
    else:
        print("No results to evaluate. Please check for errors above.")

if __name__ == "__main__":
    main()

# TODO: Future improvements:
# 1. Adapt code to work in non-Colab environments while maintaining simplicity
# 2. Add more customization options for visualizations
