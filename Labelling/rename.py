import os


def rename_images(folder_path):
    files = os.listdir(folder_path)

    counter = 1

    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            new_name = f"{counter}.jpg" 

            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_name)

            os.rename(old_file_path, new_file_path)

            print(f"Renamed: {filename} to {new_name}")

            counter += 1

# Specify the folder containing images
folder_path = r'C:\Users\darkn\Documents\Smart-Vision-Technology-FlipkartRobotics\Data\atta&flours'  # Change this to your image folder path

# Call the function
rename_images(folder_path)

print("Image renaming complete.")
