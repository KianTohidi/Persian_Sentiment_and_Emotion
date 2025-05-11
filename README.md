# Persian Sentiment and Emotion Analysis
This repository contains code for preprocessing and analyzing Persian text data for sentiment and emotion recognition tasks.

## Overview
This project provides tools and scripts for working with Persian language sentiment and emotion datasets. The code is organized in numerical order for a step-by-step workflow from data preprocessing to analysis.

## Datasets
Due to licensing restrictions, the CSV files are not included in this repository. Please download them from the following sources:
- **Sentiment Dataset**: [Persian Sentiment Analysis Dataset on Kaggle](https://www.kaggle.com/datasets/instatext/persian-sentiment-analysis-dataset)
- **Emotion Dataset**: [Arman Text Emotion Dataset on GitHub](https://github.com/Arman-Rayan-Sharif/arman-text-emotion)

## Sample Data
Due to licensing restrictions, the original CSV files are not included in this repository. However, I've created sample files to demonstrate the expected format for inputs and outputs:

- **Sample Files**: The `01_sentiment_files_samples` folder contains:
  - A sample input CSV showing the required format before processing
  - A sample output CSV showing the expected structure after processing
    
  The `02_emotion_files_samples` folder contains:
  - 2 samples input TSV showing the required format before processing
  - A sample output CSV showing the expected structure after processing
  

These samples are provided to help researchers understand the data format for reproducing the study. Please download the actual datasets from the links provided in the Datasets section.

## Code Organization
The codebase is organized in numerical order for reproducibility:
1. `01_sentiment_dataset_cleaning.py`: Cleans and preprocesses the Persian sentiment dataset + Visualizations
2. `02_emotion_dataset_cleaning_p1.py`: Cleans and preprocesses the Persian emotion dataset (Part 1) + Visualizations
3. `03_emotion_dataset_cleaning_p2.py`: Cleans and preprocesses the Persian emotion dataset (Part 2) + Visualizations
4. `04_sentiment_dataset_sampling.py`: Samples and balances the Persian sentiment dataset
5. `05_emotion_dataset_sampling.py`: Samples and balances the Persian emotion dataset

## Getting Started
### Prerequisites
- Google Colab (recommended environment)
- Python (tested with version 3.11.12)
- Pandas (tested with version 2.2.2)
- NumPy (tested with version 2.0.2)
- Matplotlib (tested with version 3.10.0)
- Seaborn (tested with version 0.13.2)

### Setup Instructions
1. Clone this repository
2. Download the datasets from the links provided above
3. Upload the code files and datasets to Google Colab
4. Follow the instructions in the code comments for each file
5. Run the scripts in numerical order

## Reproducibility
To ensure reproducibility:
1. Use Google Colab as your execution environment
2. Follow the numerical order of the code files
3. Read the comments in each file carefully, especially regarding data preparation
4. Use the same Python and package versions listed in the dependencies

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the creators of the Persian Sentiment Analysis Dataset and Arman Text Emotion Dataset for making their data available for research
- The Natural Language Processing community for developing open techniques and resources for low-resource languages
- Professor Simone Rebora, my supervisor for my thesis, who provided guidance during the work

