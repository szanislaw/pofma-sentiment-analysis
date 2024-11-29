import os
import pandas as pd
import matplotlib.pyplot as plt

folder_path = 'scraped-comments-sentiment'

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