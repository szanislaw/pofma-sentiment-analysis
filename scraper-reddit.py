import requests
import json
import pandas as pd
import os
import csv
import re

csv_file_path = 'data/pofma-related-articles-reddit.csv'
urls_df = pd.read_csv(csv_file_path)

output_dir = 'data/scraped-comments'
os.makedirs(output_dir, exist_ok=True)

counter = 1

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "", title)[:50]

def fetch_comments(comment_list, comments, link_id):
    for comment in comment_list:
        if comment['kind'] == 't1':  # t1 means it's a comment
            body = comment['data'].get('body', '[Deleted]')  # capture deleted comments
            comments.append([body])
        elif comment['kind'] == 'more':  
            more_ids = ','.join(comment['data']['children'])
            # more replies URL
            more_url = f"https://www.reddit.com/api/morechildren.json?link_id=t3_{link_id}&children={more_ids}&api_type=json"
            more_response = requests.get(more_url, headers={'User-agent': 'POFMAScraperBot/1.0 by szanislaw'})
            if more_response.status_code == 200:
                more_data = more_response.json()['json']['data']['things']
                fetch_comments(more_data, comments, link_id)

# scraping logic
for index, row in urls_df.iterrows():
    reddit_post_url = row.iloc[0] + '.json?limit=500'
    response = requests.get(reddit_post_url, headers={'User-agent': 'POFMAScraperBot/1.0 by szanislaw'})
    
    if response.status_code == 200:
        comments = []
        
        post_data = response.json()
        title = post_data[0]['data']['children'][0]['data']['title']
        sanitized_title = sanitize_filename(title)
        link_id = post_data[0]['data']['children'][0]['data']['id']
        comment_list = post_data[1]['data']['children']
        
        fetch_comments(comment_list, comments, link_id)

        file_name = f"reddit-{counter} - {sanitized_title}.csv"
        output_file_path = os.path.join(output_dir, file_name)
        
        with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Comment"])
            writer.writerows(comments)
        
        print(f"Scraping complete for '{title}'. Results saved to '{output_file_path}'.")
        counter += 1
    else:
        print(f"Failed to retrieve data for URL: {reddit_post_url}, Status Code: {response.status_code}")

print("All files processed.")
