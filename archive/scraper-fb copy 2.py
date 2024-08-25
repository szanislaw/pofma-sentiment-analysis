from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import csv

# Chrome options
chrome_options = Options()
chrome_options.debugger_address = "localhost:9222"

# Use webdriver_manager to manage ChromeDriver dynamically
webdriver_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

def get_facebook_comments(url):
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    # Manually scroll to the bottom of the page to load all comments
    scroll_to_bottom(driver)

    # Click 'View more comments' buttons to load all comments
    while True:
        try:
            load_more_button = driver.find_element(By.XPATH, "//span[contains(@class, 'x193iq5w') and contains(text(), 'View more comments')]")
            load_more_button.click()
            time.sleep(2)
        except Exception as e:
            print(f"No more 'View more comments' button found: {e}")
            break

    # Expand all "See more" links within comments
    while True:
        try:
            see_more_buttons = driver.find_elements(By.XPATH, "//div[@role='button' and @tabindex='0' and contains(text(), 'See more') or contains(text(), 'Mehr anzeigen')]")
            if not see_more_buttons:
                break
            for button in see_more_buttons:
                driver.execute_script("arguments[0].click();", button)
                time.sleep(1)
        except Exception as e:
            print(f"No more 'See more' buttons found: {e}")
            break

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    comments = []
    # Adjusting to the specified HTML structure
    comment_elements = soup.find_all('span', class_="x193iq5w xeuugli x13faqbe x1vvkbs x10flsy6 x1lliihq x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x4zkp8e x41vudc x6prxxf xvq8zen xo1l8bm xzsf02u")
    print(f"Found {len(comment_elements)} comment elements")
    for element in comment_elements:
        comment_text = element.get_text(strip=True)
        if comment_text != 'Facebook':
            comments.append(comment_text)

    return comments

def save_comments_to_csv(comments, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comment"])
        for comment in comments:
            writer.writerow([comment])

def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for the page to load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# Example usage
url = "https://www.facebook.com/yoursdp/posts/correction-noticethis-post-contains-a-false-statement-of-fact-there-is-no-rising/10158348000643455/"

comments = get_facebook_comments(url)

# Save the comments to a CSV file
save_comments_to_csv(comments, 'facebook_comments.csv')

# Print the comments (optional)
for comment in comments:
    print(comment)

driver.quit()
