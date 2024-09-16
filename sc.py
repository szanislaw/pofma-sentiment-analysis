from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

# Path to your cookies file
cookie_file_path = "coursera.json"

# Set up Chrome options
chrome_options = webdriver.ChromeOptions()

# Initialize the Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Navigate to the Coursera "My Learning" page
url = "https://www.coursera.org/my-learning?myLearningTab=IN_PROGRESS"
driver.get(url)

# Load cookies from the JSON file
with open(cookie_file_path, "r") as file:
    cookies = json.load(file)
    for cookie in cookies:
        # Remove 'sameSite' attribute if present, as it's not supported by Selenium
        if 'sameSite' in cookie:
            del cookie['sameSite']
        driver.add_cookie(cookie)

# Refresh the page to apply cookies
driver.refresh()

time.sleep(5)  # Give time for the page to load

try:
    while True:
        # Find all elements with the class 'cds-button-label' and text 'Reset Deadlines'
        reset_buttons = driver.find_elements(By.XPATH, "//span[@class='cds-button-label' and contains(text(), 'Reset deadlines')]")

        if reset_buttons:
            for button in reset_buttons:
                try:
                    button.click()
                    print("Clicked on 'Reset Deadlines'")
                    time.sleep(1)  # Add delay to avoid being detected as a bot
                except Exception as e:
                    print(f"Failed to click on 'Reset deadlines': {e}")
                    continue
        else:
            print("No 'Reset Deadlines' buttons found.")

        time.sleep(30)  # Wait before checking again

except KeyboardInterrupt:
    print("Script stopped by user.")

finally:
    driver.quit()
