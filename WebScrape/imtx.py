import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def get_images_from_google(driver, delay, max_images, text_file_path):
    def scroll_down(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    image_urls = set()
    skips = 0

    while len(image_urls) + skips < max_images:
        scroll_down(driver)

        thumbnails = driver.find_elements(By.CSS_SELECTOR, '.rg_i, .Q4LuWd')
        for img in thumbnails[len(image_urls) + skips:max_images]:
            try:
                img.click()
                time.sleep(delay)
            except:
                continue

            images = driver.find_elements(By.CSS_SELECTOR, '.r48jcc, .pT0Scc, .iPVvYb')
            for image in images:
                image_url = image.get_attribute('src')
                if image_url in image_urls:
                    skips += 1
                    break

                if image_url and 'http' in image_url:
                    image_urls.add(image_url)
                    print(f"Found {len(image_urls)}: {image_url}")

                    # Append the image URL to the text file
                    append_to_file(text_file_path, image_url)

                    if len(image_urls) >= max_images:
                        return  # Stop once we have enough images

def append_to_file(file_path, url):
    with open(file_path, "a") as f:  # Open the file in append mode
        f.write(url + "\n")  # Write the URL to the file followed by a newline

# Ask the user for the search query
search_query = input("Enter your Google Images search query: ")

# Create the 'imgs/' directory if it doesn't exist
download_path = "D:/##BTECH TOTAL/@Hackathons/Smart-Vision-Technology-FlipkartRobotics/imgs"
os.makedirs(download_path, exist_ok=True)

# Define the text file path to save URLs
text_file_path = os.path.join(download_path, "image_urls.txt")

# Create a Chrome driver
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Open the Google Images search page with the provided search query
search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
driver.get(search_url)

# Perform image scraping and save URLs to the text file
get_images_from_google(driver, 2, 1000, text_file_path)

# Close the driver instance
driver.quit()

print(f"Image URLs have been saved to {text_file_path}.")
