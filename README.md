# Persian Sentiment and Emotion Analysis
This repository contains code for preprocessing and analyzing Persian text data for sentiment and emotion recognition tasks.

## Overview
This project provides tools and scripts for working with Persian language sentiment and emotion datasets. The code is organized in numerical order for a step-by-step workflow from data preprocessing to analysis and includes multiple AI model implementations for sentiment and emotion detection.

## Datasets
Due to licensing restrictions, the CSV files are not included in this repository. Please download them from the following sources:
- **Sentiment Dataset**: [Persian Sentiment Analysis Dataset on Kaggle](https://www.kaggle.com/datasets/instatext/persian-sentiment-analysis-dataset)
- **Emotion Dataset**: [Arman Text Emotion Dataset on GitHub](https://github.com/Arman-Rayan-Sharif/arman-text-emotion)

## Sample Data
Due to licensing restrictions, the original CSV files are not included in this repository. However, I've created sample files to demonstrate the expected format for inputs and outputs:
- **Sample Files**:
- The `01_sentiment_files_samples` folder contains:
  - A sample input CSV showing the required format before processing
  - A sample output CSV showing the expected structure after processing
    
  The `02_emotion_files_samples` folder contains:
  - 2 samples input TSV showing the required format before processing
  - A sample output CSV showing the expected structure after processing
  
These samples are provided to help researchers understand the data format for reproducing the study. Please download the actual datasets from the links provided in the Datasets section.

## Prompts
The `03_prompts` folder contains:
- `prompts/sentiment_analysis_prompt.txt`: Complete prompt used for sentiment analysis task
- `prompts/emotion_detection_prompt.txt`: Complete prompt used for emotion detection task

## Code Organization
The codebase is organized in numerical order for reproducibility:

### Data Preprocessing:
1. `01_sentiment_dataset_cleaning.py`: Cleans and preprocesses the Persian sentiment dataset + Visualizations
2. `02_emotion_dataset_cleaning_p1.py`: Cleans and preprocesses the Persian emotion dataset (Part 1) + Visualizations
3. `03_emotion_dataset_cleaning_p2.py`: Cleans and preprocesses the Persian emotion dataset (Part 2) + Visualizations
4. `04_sentiment_dataset_sampling.py`: Samples and balances the Persian sentiment dataset
5. `05_emotion_dataset_sampling.py`: Samples and balances the Persian emotion dataset

### AI Model Implementations:

#### Claude 3.7 Sonnet:
6. `06_claude_sentiment_analysis.py`: Implementation of sentiment analysis using Claude 3.7 Sonnet
7. `07_claude_emotion_detection.py`: Implementation of emotion detection using Claude 3.7 Sonnet

#### DeepSeek-V3:
8. `08_deepseek_sentiment_analysis.py`: Implementation of sentiment analysis using DeepSeek-V3
9. `09_deepseek_emotion_detection.py`: Implementation of emotion detection using DeepSeek-V3

#### Gemini 2.0 Flash:
10. `10_gemini_sentiment_analysis.py`: Implementation of sentiment analysis using Gemini 2.0 Flash
11. `11_gemini_emotion_detection.py`: Implementation of emotion detection using Gemini 2.0 Flash

#### GPT-4o:
12. `12_gpt4o_sentiment_analysis.py`: Implementation of sentiment analysis using GPT-4o
13. `13_gpt4o_emotion_detection.py`: Implementation of emotion detection using GPT-4o

### Statistical Analysis and Evaluation:
14. `14_sentiment_statistical_analysis.py`: Bootstrap confidence intervals and McNemar's test for sentiment analysis model comparison
15. `15_emotion_statistical_analysis.py`: Bootstrap confidence intervals and McNemar's test for emotion detection model comparison

## Getting Started
### Prerequisites
- Google Colab (recommended environment)
- Python (tested with version 3.11.12)
- Pandas (tested with version 2.2.2)
- NumPy (tested with version 2.0.2)
- Matplotlib (tested with version 3.10.0)
- Seaborn (tested with version 0.13.2)
- API keys for Claude, DeepSeek, Gemini, and GPT-4o

### Setup Instructions
1. Clone this repository
2. Download the datasets from the links provided above
3. Upload the code files and datasets to Google Colab
4. Set up your API keys as environment variables
5. Follow the instructions in the code comments for each file
6. Run the scripts in numerical order

## Reproducibility
To ensure reproducibility:
1. Use Google Colab as your execution environment
2. Follow the numerical order of the code files
3. Read the docstrings and comments in each file carefully, especially regarding data preparation
4. Use the same Python and package versions listed in the dependencies

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the creators of the Persian Sentiment Analysis Dataset and Arman Text Emotion Dataset for making their data available for research
- The Natural Language Processing community for developing open techniques and resources for low-resource languages
- Professor Simone Rebora, my supervisor for my thesis, who provided guidance during the work
