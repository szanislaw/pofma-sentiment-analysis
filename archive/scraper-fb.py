from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import os
import json
import pandas as pd
import time

def load_cookies(driver, cookies_file):
    with open(cookies_file, 'r') as f:
        cookies = json.load(f)
    for cookie in cookies:
        if 'domain' in cookie and cookie['domain'].startswith('.'):
            cookie['domain'] = cookie['domain'][1:]  # Remove leading dot for compatibility
        
        if 'sameSite' in cookie:
            if cookie['sameSite'] not in ['Strict', 'Lax', 'None']:
                cookie['sameSite'] = 'Lax'  # Set a default value if unexpected value
        else:
            cookie['sameSite'] = 'Lax'  # Set a default value if sameSite is missing
        
        try:
            driver.add_cookie(cookie)
        except Exception as e:
            print(f"Could not add cookie: {cookie['name']} due to {e}")

# See More
def click_see_more_buttons(driver):
    see_more_buttons = driver.find_elements(By.XPATH, '//div[@role="button" and contains(text(), "See more")]')
    for button in see_more_buttons:
        try:
            button.click()
            time.sleep(1) 
        except Exception as e:
            print(f"Could not click 'See more' button due to {e}")

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# chrome_options.add_argument("--headless")  # Uncomment this line if you want to run in headless mode

chrome_prefs = {
    "profile.default_content_setting_values.notifications": 2  # 1-Allow
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

webdriver_service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

# Replace 'your_facebook_post_url' with the actual URL
url = "https://www.facebook.com/yoursdp/posts/correction-noticethis-post-contains-a-false-statement-of-fact-there-is-no-rising/10158348000643455/"
driver.get(url)

# Load cookies to stay logged in
cookies_file = 'fbcookie.pkl'
load_cookies(driver, cookies_file)

# Refresh the page to apply cookies
driver.get(url)

# Optional: Maximize the window
driver.maximize_window()

# Wait until the necessary elements are loaded (you might need to adjust this wait time)
driver.implicitly_wait(10)

# Scroll and load more comments
previous_height = driver.execute_script("return document.body.scrollHeight")
scroll_pause_time = 3  # Adjust the pause time as needed

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)  # Wait for the page to load
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    # Click "See more" buttons after scrolling
    click_see_more_buttons(driver)
    
    if new_height == previous_height:  # If no more content is loaded, break the loop
        break
    
    previous_height = new_height

# Find all the div elements with the specific attributes
comments = driver.find_elements(By.XPATH, '//div[@dir="auto" and @style="text-align: start;"]')

# Extract the text from each found element and store it in a list
scraped_comments = [comment.text for comment in comments]

# Prepare the folder and file path for saving
output_folder = 'scraped-comments'
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, 'fb_comments.csv')

# Save the scraped comments to a CSV file
df = pd.DataFrame(scraped_comments, columns=['Comment'])
df.to_csv(output_file, index=False, encoding='utf-8')

# Remember to close the driver when done
driver.quit()
