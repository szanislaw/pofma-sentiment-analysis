import os
import pandas as pd
from transformers import pipeline

# Initialize the sentiment analysis pipeline with the specified model
sentiment_analyzer = pipeline('sentiment-analysis', model='MarieAngeA13/Sentiment-Analysis-BERT')

# Path to the folder containing the CSV files
folder_path = 'reddit_comments'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Process each CSV file
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Perform sentiment analysis on each comment
    df['sentiment'] = df['Comment'].apply(lambda x: sentiment_analyzer(x.strip())[0]['label'])
    
    # Save the results to a new CSV file
    output_file_path = os.path.join(folder_path, f'processed_{csv_file}')
    df.to_csv(output_file_path, index=False)
    
    print(f"Sentiment analysis complete for {csv_file}. Results saved to '{output_file_path}'.")

print("All files processed.")
