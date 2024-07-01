from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import csv

# Configure Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

chrome_prefs = {
    "profile.default_content_setting_values.notifications": 2  # 1-Allow, 2-Block
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

webdriver_service = Service("chromedriver-linux64/chromedriver")  # Change this to your actual path
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

# Navigate to the Reddit post
reddit_post_url = "https://www.reddit.com/r/singapore/comments/15dbudo/pofma_and_selective_enforcement_a_matter_of/"
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
        comments.append(comment_text)

# Close the driver
driver.quit()

# Print the comments
for comment in comments:
    print(comment)

# Save the comments to a CSV file
csv_file_path = "comments.csv"
with open(csv_file_path, "w", encoding="utf-8", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Comment"])
    for comment in comments:
        writer.writerow([comment])
