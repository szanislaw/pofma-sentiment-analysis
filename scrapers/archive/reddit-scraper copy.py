from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import csv
import pandas as pd

# Load the CSV file with URLs
csv_file_path = 'data/pofma-related-articles-reddit.csv'
reddit_links_df = pd.read_csv(csv_file_path)
urls = reddit_links_df.iloc[:, 0].tolist()

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

chrome_prefs = {
    "profile.default_content_setting_values.notifications": 2  # 1-Allow, 2-Block
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

webdriver_service = Service("chromedriver-linux64/chromedriver")  # Change this to your actual path
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

all_comments = []

for reddit_post_url in urls:
    driver.get(reddit_post_url)
    time.sleep(5)  # Wait for the page to load

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Scrape comments
    comments = soup.find_all('div', {'data-test-id': 'comment'})
    for comment in comments:
        comment_text = comment.find('p').text if comment.find('p') else 'No text'
        all_comments.append({
            'url': reddit_post_url,
            'comment': comment_text
        })

# Save comments to CSV
output_csv_file = 'reddit_comments.csv'
keys = all_comments[0].keys()
with open(output_csv_file, 'w', newline='', encoding='utf-8') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_comments)

driver.quit()
