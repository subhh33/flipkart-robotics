import requests

# URL of the image to download
image_url = "https://static.vecteezy.com/system/resources/thumbnails/024/646/930/small_2x/ai-generated-stray-cat-in-danger-background-animal-background-photo.jpg"

# Function to download the image
def download_image(url, file_name):
    try:
        # Send a GET request to the image URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Write the image content to a file
        with open(file_name, "wb") as file:
            file.write(response.content)
        
        print("Image downloaded successfully.")
    except Exception as e:
        print("Failed to download image:", e)

# Call the function to download the image
download_image(image_url, "cat_image.jpg")
