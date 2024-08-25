import os
import pandas as pd
from transformers import pipeline

# Initialize the sentiment analysis pipeline with the specified model
sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/bertweet-base-sentiment-analysis')

# Path to the folder containing the original CSV files
input_folder_path = 'scraped-comments/reddit'

# Path to the folder where sentiment-analyzed files will be saved
output_folder_path = 'scraped-comments-sentiment/reddit'

# Create the output directory if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Function to split text into chunks of max_length
def split_text_into_chunks(text, max_length=128):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# List all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]

# Process each CSV file
for csv_file in csv_files:
    file_path = os.path.join(input_folder_path, csv_file)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Perform sentiment analysis on each comment
    if 'Comment' in df.columns:
        sentiments = []
        for comment in df['Comment']:
            chunks = split_text_into_chunks(comment.strip())
            chunk_sentiments = [sentiment_analyzer(chunk)[0]['label'] for chunk in chunks]
            # You can aggregate chunk sentiments if needed (e.g., majority vote, averaging, etc.)
            sentiment = chunk_sentiments[0]  # Example: take the sentiment of the first chunk
            sentiments.append(sentiment)
        
        df['sentiment'] = sentiments
    
        # Save the results to a new CSV file in the output folder
        output_file_path = os.path.join(output_folder_path, f'processed_{csv_file}')
        df.to_csv(output_file_path, index=False)
        
        print(f"Sentiment analysis complete for {csv_file}. Results saved to '{output_file_path}'.")
    else:
        print(f"Column 'Comment' not found in {csv_file}. Skipping file.")

print("All files processed.")
