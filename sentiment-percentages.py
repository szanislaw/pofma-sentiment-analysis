import pandas as pd
import os

def calculate_sentiment_percentages(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Calculate the number of each sentiment type
    sentiment_counts = df['sentiment'].value_counts()
    
    # Calculate total comments
    total_comments = sentiment_counts.sum()
    
    # Calculate percentages
    percentages = {
        'positive': (sentiment_counts.get('positive', 0) / total_comments) * 100,
        'neutral': (sentiment_counts.get('neutral', 0) / total_comments) * 100,
        'negative': (sentiment_counts.get('negative', 0) / total_comments) * 100
    }
    
    return percentages

def process_all_csv_files(folder_path):
    all_percentages = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            percentages = calculate_sentiment_percentages(file_path)
            all_percentages.append(percentages)
    
    # Convert the list of dictionaries to a DataFrame for better visualization
    percentages_df = pd.DataFrame(all_percentages)
    
    # Calculate the average percentages
    average_percentages = percentages_df.mean().to_dict()
    average_percentages['filename'] = 'avg'
    
    # Append the average to the DataFrame
    percentages_df = pd.concat([percentages_df, pd.DataFrame([average_percentages])], ignore_index=True)
    
    return percentages_df

# Folder containing the CSV files
folder_path = 'data/reddit_comments/sentiment-comments'

# Process all CSV files and get the percentages
sentiment_percentages_df = process_all_csv_files(folder_path)

# Display the DataFrame
print(sentiment_percentages_df)

# Save the results to a new CSV file
output_file_path = 'sentiment_percentages.csv'
sentiment_percentages_df.to_csv(output_file_path, index=False)
