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
output_dir = 'scraped-comments'
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for numbering the files
counter = 1

# Function to sanitize the title for use as a filename
def sanitize_filename(title):
    # Remove invalid characters and limit length
    return re.sub(r'[\\/*?:"<>|]', "", title)[:50]

# Function to recursively fetch comments, including "more" comments
def fetch_comments(comment_list, comments, link_id):
    for comment in comment_list:
        if comment['kind'] == 't1':  # t1 means it's a comment
            body = comment['data'].get('body', '[Deleted]')  # Capture deleted comments too
            comments.append([body])
        elif comment['kind'] == 'more':  # More comments to fetch
            more_ids = ','.join(comment['data']['children'])
            # This URL fetches hidden comments ("more replies")
            more_url = f"https://www.reddit.com/api/morechildren.json?link_id=t3_{link_id}&children={more_ids}&api_type=json"
            more_response = requests.get(more_url, headers={'User-agent': 'POFMAScraperBot/1.0 by szanislaw'})
            if more_response.status_code == 200:
                more_data = more_response.json()['json']['data']['things']
                # Recursively fetch more comments from the 'more' data
                fetch_comments(more_data, comments, link_id)

# Iterate over each URL and scrape comments
for index, row in urls_df.iterrows():
    reddit_post_url = row.iloc[0] + '.json?limit=500'
    response = requests.get(reddit_post_url, headers={'User-agent': 'POFMAScraperBot/1.0 by szanislaw'})
    
    if response.status_code == 200:
        post_data = response.json()

        # Extract the title and sanitize it
        title = post_data[0]['data']['children'][0]['data']['title']
        sanitized_title = sanitize_filename(title)

        comments = []
        # Get the post's link ID, required to fetch more replies
        link_id = post_data[0]['data']['children'][0]['data']['id']

        # Navigate through the JSON structure to find comments
        comment_list = post_data[1]['data']['children']
        fetch_comments(comment_list, comments, link_id)

        # Format the output filename with a numbered prefix and the sanitized title
        file_name = f"reddit-{counter} - {sanitized_title}.csv"
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
