from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import csv
import pickle
from webdriver_manager.chrome import ChromeDriverManager
import json

# Chrome config
chrome_options = Options()
# chrome_options.add_argument("--headless")  
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

chrome_prefs = {
    "profile.default_content_setting_values.notifications": 2  # 1-Allow, 2-Block
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

# Use webdriver_manager to manage ChromeDriver dynamically
webdriver_service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

def load_cookies(driver, cookies_file_path):
    driver.get("https://www.facebook.com/")
    with open(cookies_file_path, "r") as cookiesfile:
        cookies = json.load(cookiesfile)  # Load the cookies using json
        for cookie in cookies:
            # Remove 'sameSite' if it's not valid
            if 'sameSite' in cookie and cookie['sameSite'] not in ["Strict", "Lax", "None"]:
                del cookie['sameSite']
            driver.add_cookie(cookie)
    driver.refresh()  # Refresh the page to ensure cookies are loaded
    
def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for the page to load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
def expand_see_more(driver):
    while True:
        try:
            see_more_buttons = driver.find_elements(By.XPATH, "//div[@role='button' and @tabindex='0' and contains(text(), 'See more')]")
            if not see_more_buttons:
                break
            for button in see_more_buttons:
                driver.execute_script("arguments[0].click();", button)
                time.sleep(1)  # Small delay to ensure the page is updated
        except Exception as e:
            print(f"No more 'See more' buttons found or an error occurred: {e}")
            break


def get_facebook_comments(url, cookies_file_path):
    load_cookies(driver, cookies_file_path)
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    # Scroll to the bottom to load all comments
    scroll_to_bottom(driver)

    # Expand all "See more" links within comments
    expand_see_more(driver)

    # Find all div elements with aria-label containing 'Comment'
    comment_elements = driver.find_elements(By.XPATH, "//div[contains(@aria-label, 'Comment')]")
    print(f"Found {len(comment_elements)} comment elements")
    
    comments = []
    for element in comment_elements:
        comment_text = element.text
        if comment_text:
            comments.append(comment_text)

    return comments


def save_comments_to_csv(comments, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comment"])
        for comment in comments:
            writer.writerow([comment])

# Example usage
url = "https://www.facebook.com/yoursdp/posts/correction-noticethis-post-contains-a-false-statement-of-fact-there-is-no-rising/10158348000643455/"
cookies_file_path = "fbcookie.pkl"  # Ensure you have your cookies stored here

comments = get_facebook_comments(url, cookies_file_path)

# Save the comments to a CSV file
save_comments_to_csv(comments, 'facebook_comments.csv')

# Print the comments (optional)
for comment in comments:
    print(comment)

driver.quit()
