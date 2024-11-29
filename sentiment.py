import os
import pandas as pd
from transformers import pipeline

sentiment_analyzer = pipeline('sentiment-analysis', model='finiteautomata/bertweet-base-sentiment-analysis', device=0)

input_folder_path = 'data/scraped-comments'
output_folder_path = 'data/scraped-comments-sentiment'
os.makedirs(output_folder_path, exist_ok=True)

def split_text_into_chunks(text, max_length=128):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(input_folder_path, csv_file)
    
    df = pd.read_csv(file_path)
    
    # sentiment analysis
    if 'Comment' in df.columns:
        sentiments = []
        for comment in df['Comment']:
            chunks = split_text_into_chunks(comment.strip())
            chunk_sentiments = [sentiment_analyzer(chunk)[0]['label'] for chunk in chunks]
            sentiment = chunk_sentiments[0] 
            sentiments.append(sentiment)
        
        df['sentiment'] = sentiments
    
        output_file_path = os.path.join(output_folder_path, f'processed_{csv_file}')
        df.to_csv(output_file_path, index=False)
        
        print(f"Sentiment analysis complete for {csv_file}. Results saved to '{output_file_path}'.")
    else:
        print(f"Column 'Comment' not found in {csv_file}. Skipping file.")

print("All files processed.\n")

folder_path = 'data/scraped-comments-sentiment'

total_sentiment_counts = {'NEG': 0, 'NEU': 0, 'POS': 0}

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        sentiment_counts = df['sentiment'].value_counts()
        
        for sentiment, count in sentiment_counts.items():
            if sentiment in total_sentiment_counts:
                total_sentiment_counts[sentiment] += count

print("Cumulative Sentiment Counts:")
print(f"NEG: {total_sentiment_counts['NEG']}")
print(f"NEU: {total_sentiment_counts['NEU']}")
print(f"POS: {total_sentiment_counts['POS']}")