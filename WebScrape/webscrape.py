import requests
from bs4 import BeautifulSoup
import os

# Create a folder to save the images
if not os.path.exists('grocery_images'):
    os.makedirs('grocery_images')

# Function to download images from a URL
def download_image(img_url, img_name):
    response = requests.get(img_url)
    if response.status_code == 200:
        with open(f'grocery_images/{img_name}.jpg', 'wb') as file:
            file.write(response.content)

# Flipkart URL for groceries
url = 'https://www.google.com/search?sca_esv=a5db118e05c2ad26&sxsrf=ADLYWIK5IEstvPjW-3QYeSPWDiU7HCcYzA:1728212082465&q=salt+packet&udm=2&fbs=AEQNm0CvspUPonaF8UH5s_LBD3JPX4RSeMPt9v8oIaeGMh2T2D1DyqhnuPxLgMgOaYPYX7OtOF4SxbM4YPsyWUMdeXRPnkQc3caC_NEMjyGZlBqX7YDVSc-lk14rE2h7j-ln6ORWjT4WxqVC6FS82YpEwEqqnkJJKpHqKGrk5ZhbNsOcE3i19GRoFANVfwr_gZS3oWcL17KMyupN4i8_p5OTUvqC1CSN_g&sa=X&sqi=2&ved=2ahUKEwj_iZbay_mIAxWpRmcHHeDUFVEQtKgLegQIExAB&biw=1182&bih=541&dpr=1.63'

# Request the webpage content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the images (usually in <img> tags with a specific class)
images = soup.find_all('img', {'class': '_396cs4'})

# Loop through the images and download them
for idx, img in enumerate(images):
    img_url = img['src']  # Image URL
    download_image(img_url, f'grocery_item_{idx}')  # Download and save
    print(f"Downloaded: grocery_item_{idx}.jpg")

print("Image scraping completed!")
