import requests
import json
import pandas as pd
import os
import csv
import re

# Read the CSV file with Reddit post URLs
csv_file_path = 'data/pofma-related-articles-reddit.csv'
urls_df = pd.read_csv(csv_file_path)

# Update the destination path
output_dir = 'scraped-comments/reddit'
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for numbering the files
counter = 1

# Function to sanitize the title for use as a filename
def sanitize_filename(title):
    # Remove invalid characters and limit length
    return re.sub(r'[\\/*?:"<>|]', "", title)[:50]

# Iterate over each URL and scrape comments
for index, row in urls_df.iterrows():
    reddit_post_url = row.iloc[0] + '.json'
    response = requests.get(reddit_post_url, headers={'User-agent': 'your_user_agent'})
    
    if response.status_code == 200:
        post_data = response.json()

        # Extract the title and sanitize it
        title = post_data[0]['data']['children'][0]['data']['title']
        sanitized_title = sanitize_filename(title)

        comments = []
        # Navigate through the JSON structure to find comments
        for comment in post_data[1]['data']['children']:
            if 'body' in comment['data']:
                comments.append([comment['data']['body']])

        # Format the output filename with a numbered prefix and the sanitized title
        file_name = f"{counter} - {sanitized_title}.csv"
        output_file_path = os.path.join(output_dir, file_name)
        
        # Save the comments to the CSV file
        with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Comment"])
            writer.writerows(comments)
        
        print(f"Scraping complete for '{title}'. Results saved to '{output_file_path}'.")

        # Increment the counter for the next file
        counter += 1
    else:
        print(f"Failed to retrieve data for URL: {reddit_post_url}, Status Code: {response.status_code}")

print("All files processed.")
