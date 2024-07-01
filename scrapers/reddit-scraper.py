from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import csv
import pandas as pd
import os

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

chrome_prefs = {
    "profile.default_content_setting_values.notifications": 2  # 1-Allow, 2-Block
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

webdriver_service = Service("chromedriver-linux64/chromedriver")  # Change this to your actual path

# Read the CSV file with Reddit post URLs
csv_file_path = 'data/pofma-related-articles-reddit.csv'
urls_df = pd.read_csv(csv_file_path)

# Create a directory to save the comments CSV files
output_dir = 'data/reddit_comments'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each URL and scrape comments
for index, row in urls_df.iterrows():
    reddit_post_url = row[0]
    driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)
    driver.get(reddit_post_url)

    # Wait for the page to load
    time.sleep(5)

    # Scroll to the bottom to load all comments (you may need to adjust the scrolling logic)
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Get the page source and parse it with BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Extract comments
    comments = []
    comment_elements = soup.find_all("div", {"id": "-post-rtjson-content"})

    for comment in comment_elements:
        p_tags = comment.find_all("p")
        for p in p_tags:
            comment_text = p.get_text()
            comments.append([comment_text])  # Store comments as lists for CSV writing

    # Close the driver
    driver.quit()

    # Save comments to a CSV file named after the Reddit post's unique identifier
    post_id = reddit_post_url.split("/")[-3]
    output_file_path = os.path.join(output_dir, f"{post_id}_comments.csv")
    
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comment"])
        writer.writerows(comments)

print("Scraping completed.")
