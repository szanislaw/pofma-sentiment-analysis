from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os
import pickle

# Function to load cookies
def load_cookies(driver, cookies_file):
    cookies = pickle.load(open(cookies_file, "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)

# Use the full path to chromedriver.exe
driver_path = os.path.abspath('chromedriver.exe')
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# Replace 'your_facebook_post_url' with the actual URL
url = "https://www.facebook.com/yoursdp/posts/correction-noticethis-post-contains-a-false-statement-of-fact-there-is-no-rising/10158348000643455/"
driver.get(url)

# Load cookies to stay logged in
cookies_file = 'fb-cookies/facebook_cookies.pkl'
load_cookies(driver, cookies_file)

# Refresh the page to apply cookies
driver.get(url)

# Optional: Maximize the window
driver.maximize_window()

# Wait until the necessary elements are loaded (you might need to adjust this wait time)
driver.implicitly_wait(10)

# Find all the div elements with the specific attributes
comments = driver.find_elements_by_xpath('//div[@dir="auto" and @style="text-align: start;"]')

# Extract and print the text from each found element
scraped_comments = []
for comment in comments:
    text = comment.text
    print(text)
    scraped_comments.append(text)

# Optionally, save the scraped comments to a file
output_file = 'scraped-comments-sentiment/reddit/fb_comments.txt'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    for comment in scraped_comments:
        f.write(comment + "\n")

# Remember to close the driver when done
driver.quit()
