import os
import pandas as pd

def natural_key(filename):
    return int(filename.split('.')[0])
def create_csv_with_images(folder_path, csv_file_path):
    image_names = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_names.append(filename)
     
    sorted_image_names = sorted(image_names, key=natural_key)
    # print(sorted_image_names)
    df = pd.DataFrame(sorted_image_names, columns=['image_name'])
    df['brand_name'] = ''
    df['item_name'] = ''
    df['MRP'] = ''
    df['pack_size'] = ''

    df.to_csv(csv_file_path, index=False)

folder_path = r'D:\##BTECH TOTAL\@Hackathons\Smart-Vision-Technology-FlipkartRobotics\Data\rice'  
csv_file_path = r'D:\##BTECH TOTAL\@Hackathons\Smart-Vision-Technology-FlipkartRobotics\Labelling\l.csv'  

create_csv_with_images(folder_path, csv_file_path)

print(f"CSV file created at {csv_file_path} with image names from {folder_path}.")
