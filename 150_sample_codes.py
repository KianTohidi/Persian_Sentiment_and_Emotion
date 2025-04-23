# Step 1: Uploading each dataset
from google.colab import files
uploaded = files.upload()

# Step 2: Import libraries
import pandas as pd

# Step 3: Load the dataset using utf-8-sig encoding
file_name = list(uploaded.keys())[0]  # Automatically use uploaded file name
df = pd.read_csv(file_name, encoding='utf-8-sig')

# Step 4: Check the first few rows (to understand structure)
df.head()

# Step 5: Sample 150 random rows
sample_df = df.sample(n=150) #  get 150 completely random rows

# Step 6: Display the sample
sample_df.head(10)  # just shows first 10, but you can scroll through all 150


# Step 7: Save the sample to a new CSV
sample_df.to_csv('sample_150.csv', index=False, encoding='utf-8-sig')

# Download the sampled file
files.download('sample_150.csv')
