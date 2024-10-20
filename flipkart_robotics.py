from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32, device_map="auto"
)

# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

"""##Preprocessing"""

import cv2
import numpy as np

def calculate_luminance(image):
    """
    Calculate the luminance of an image.

    :param image: Input image (numpy array in BGR format)
    :return: Average luminance value
    """
    # Convert image to float to prevent overflow/underflow
    image_float = image.astype(np.float32)

    # Split into B, G, R channels
    B, G, R = cv2.split(image_float)

    # Calculate luminance using the formula
    luminance = 0.114 * B + 0.587 * G + 0.299 * R

    # Compute the average luminance
    average_luminance = np.mean(luminance)

    return average_luminance

def auto_adjust_brightness(image, target_luminance=150, adjustment_factor=1.5):
    """
    Automatically adjust the brightness of an image based on its luminance.

    :param image: Input image (numpy array in BGR format)
    :param target_luminance: Desired average luminance (default: 130)
    :param adjustment_factor: Factor by which to increase brightness if needed (default: 1.2)
    :return: Brightness-adjusted image
    """
    current_luminance = calculate_luminance(image)
    print(f"Current Average Luminance: {current_luminance:.2f}")

    if current_luminance < target_luminance:
        # Calculate the required scaling factor to reach target luminance
        scaling_factor = target_luminance / (current_luminance + 1e-5)  # Add epsilon to prevent division by zero
        scaling_factor = min(scaling_factor, adjustment_factor)  # Limit the scaling factor

        print(f"Image is dark. Scaling brightness by a factor of {scaling_factor:.2f}")

        # Adjust brightness using scaling
        adjusted_image = cv2.convertScaleAbs(image, alpha=scaling_factor, beta=0)

        # Ensure that scaling does not overshoot the target
        adjusted_luminance = calculate_luminance(adjusted_image)
        print(f"Adjusted Average Luminance: {adjusted_luminance:.2f}")

        return adjusted_image
    else:
        print("Image brightness is adequate. No adjustment needed.")
        return image.copy()



def color_balance_gray_world(image):
    """
    Adjusts the color balance of an image using the Gray World Assumption.

    :param image: Input image (numpy array in BGR format)
    :return: Color-balanced image
    """
    # Split the image into B, G, R channels
    B, G, R = cv2.split(image.astype(np.float32))

    # Calculate the average of each channel
    avgB = np.mean(B)
    avgG = np.mean(G)
    avgR = np.mean(R)

    print(f"Average B: {avgB:.2f}, Average G: {avgG:.2f}, Average R: {avgR:.2f}")

    # Calculate the overall average
    avgGray = (avgB + avgG + avgR) / 3

    # Calculate scaling factors
    scaleB = avgGray / (avgB + 1e-5)
    scaleG = avgGray / (avgG + 1e-5)
    scaleR = avgGray / (avgR + 1e-5)

    print(f"Scaling Factors -> B: {scaleB:.2f}, G: {scaleG:.2f}, R: {scaleR:.2f}")

    # Apply scaling factors
    B = B * scaleB
    G = G * scaleG
    R = R * scaleR

    # Merge the channels back
    balanced_image = cv2.merge([B, G, R])

    # Clip the values to [0, 255] and convert to uint8
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)

    # Optionally, verify the new averages
    new_avgB = np.mean(balanced_image[:, :, 0])
    new_avgG = np.mean(balanced_image[:, :, 1])
    new_avgR = np.mean(balanced_image[:, :, 2])

    print(f"New Average B: {new_avgB:.2f}, New Average G: {new_avgG:.2f}, New Average R: {new_avgR:.2f}")

    return balanced_image



def remove_noise_bilateral(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Removes noise from an image using Bilateral Filtering.

    :param image: Input image (numpy array in BGR format)
    :param diameter: Diameter of each pixel neighborhood. Larger values result in more smoothing.
    :param sigma_color: Filter sigma in the color space. Larger values mean that farther colors within the pixel neighborhood
                        will be mixed together.
    :param sigma_space: Filter sigma in the coordinate space. Larger values mean that farther pixels will influence each other.
    :return: Denoised image
    """
    denoised_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return denoised_image

def scale_image(image, target_size=(640, 640)):
    # Resize the image to the target size
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def correct_skew(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to create a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect edges in the binary image
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        # Calculate the angle of the detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        # Calculate the median angle
        median_angle = np.median(angles)

        # Rotate the image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return corrected_image

    return image  # Return the original image if no lines are detected


def enhance_features(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel back with A and B channels
    enhanced_lab = cv2.merge((cl, a, b))

    # Convert LAB back to BGR color space
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def convert_to_grayscale(image):
    # Convert the image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# def apply_morphological_operations(image):
#     # Define a kernel for morphological operations
#     kernel = np.ones((5, 5), np.uint8)

#     # Apply erosion followed by dilation (closing)
#     return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# def apply_adaptive_threshold(image):
#     # Apply adaptive thresholding to binarize the image
#     return cv2.adaptiveThreshold(image, 255,
#                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                   cv2.THRESH_BINARY,
#                                   11, 2)



def preprocess_image(image, target_luminance=130, adjustment_factor=1.2):
    """
    Preprocess the image by automatically adjusting brightness.

    :param image: Input image (numpy array in BGR format)
    :param target_luminance: Desired average luminance (default: 130)
    :param adjustment_factor: Maximum scaling factor for brightness adjustment (default: 1.2)
    :return: Preprocessed image
    """
    image = auto_adjust_brightness(image, target_luminance, adjustment_factor)

    image = color_balance_gray_world(image)

    image = remove_noise_bilateral(image)

    image = scale_image(image)

    image = correct_skew(image)

    image = enhance_features(image)

    # image = convert_to_grayscale(image)


    # Future preprocessing steps can be added here
    # e.g., image = adjust_color_balance(image)
    #       image = correct_skew(image)
    #       ...

    return image

def main():
    # Read the input image
    image_path = 'assets/vegetables.jpg'  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found or unable to read.")
        return

    # Preprocess the image
    preprocessed_image = preprocess_image(image, target_luminance=150, adjustment_factor=1.5)

    # # Display the original and preprocessed images
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Brightness Adjusted Image', preprocessed_image)

    # # Wait until a key is pressed and then close the windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optionally, save the preprocessed image
    output_path = 'assets/processed_image.jpg'
    cv2.imwrite(output_path, preprocessed_image)
    print(f"Preprocessed image saved to {output_path}")

if _name_ == "_main_":
    main()

"""##Objection Detection"""

from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO('Models/model2(18).pt')  # Replace with your actual model path

# Perform prediction on the input image
results = model.predict('output_image_with_boxes.jpg', conf=0.25)

# Load the original image using OpenCV
img = cv2.imread('output_image_with_boxes.jpg')

# Retrieve class names from the model
class_names = model.names  # Dictionary: {0: 'class0', 1: 'class1', ...}

# The results object contains the boxes, class labels, and confidence scores
for result in results:
    # Extract bounding boxes, class labels, and confidence scores
    boxes = result.boxes.xyxy  # Bounding box coordinates in (x_min, y_min, x_max, y_max) format
    confidences = result.boxes.conf  # Confidence scores
    class_ids = result.boxes.cls  # Class labels (indices)

    print(len(boxes))
    # Iterate through each detected object
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = map(int, boxes[i])  # Convert coordinates to integers
        confidence = confidences[i].item()  # Convert tensor to float
        class_id = int(class_ids[i].item())  # Convert tensor to int

        # Draw bounding box
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Add a label with the class name and confidence
        label = f"{class_names[class_id]}: {confidence:.2f}"
        # # Calculate text size to create a filled rectangle as background for text
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
        cv2.putText(img, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Crop the image using bounding box coordinates
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Save or display the cropped image
        cropped_img_filename = f"cropped_{class_names[class_id]}_{i}.jpg"
        cv2.imwrite(cropped_img_filename, cropped_img)
        print(f"Saved cropped image: {cropped_img_filename}")

# Display the image with bounding boxes (Note: This may not work in some environments like Jupyter notebooks)
cv2.imshow('Image with Bounding Boxes', img)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

"""# Brand Name, Pack Size, and Product Description Extraction Model

To automatically extract key product details such as Brand Name, Pack Size, and Product Description from packaging material images using a PyTorch-based model.

## Features
The model extracts the following information from an input image of packaging material:
1. **Brand Name**: The name of the brand featured on the product packaging.
2. **Pack Size**: The size of the product's packaging, typically given in terms of weight or volume (e.g., 500g, 1L).
3. **Product Description**: A textual description of the product from the packaging (e.g., product type, key ingredients, etc.).

If any attribute cannot be detected, the model will return `'N/A'` for that attribute.

"""

import torch
import concurrent.futures

# Function to prepare inputs and run inference with timeout
def run_wout_ocr(image, timeout=60):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,  # Your image object
            },
            {
                "type": "text",
                "text": f"""Can you extract the details from the packaging material given in the image? Specifically, focus on
                extracting the following attributes:
                1. Brand Name
                2. Pack Size (e.g., weight, volume)
                3. Product Description

                The expected format is:
                Brand Name: {{brand_name if detected else 'N/A'}}
                Pack Size: {{pack_size if detected else 'N/A'}}
                Product Description: {{product_description if detected else 'N/A'}}

                If any attribute cannot be detected, strictly return 'N/A' for that attribute.""",
            },
        ],
    }
]


    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    def model_inference():
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

    try:
        # Use concurrent futures for timeout handling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(model_inference)
            output = future.result(timeout=timeout)  # timeout in seconds
            return output

    except concurrent.futures.TimeoutError:
        print(f"Inference timed out after {timeout} seconds.")
        return {"error": "Inference timed out"}
    except Exception as e:
        print(f"Inference failed: {e}")
        return {"error": f"Inference failed: {str(e)}"}

import os
from PIL import Image

# Define the path to your image folder
# image_folder = '/content/drive/MyDrive/#QWN MODEL/Dal_Pulses'
image_folder = '/content/'
# Loop through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") :  # Add other extensions if needed
        image_path = os.path.join(image_folder, filename)
        # print(f"Processing image: {filename}")


        # Open the image
        image = Image.open(image_path)

        # Pass the image to the run_inference function to extract text
        # text = ocr_paddle(image_path)
        # print(text)
        extracted_text = run_wout_ocr(image)

        # Print the extracted text
        print(f"{extracted_text}")
        print("------------------------------------")

"""# **ocr**"""



# from paddleocr import PaddleOCR

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True)

def ocr_paddle(image_path):
    # Perform OCR on the image
    img = ocr.ocr(image_path)

    # Check if img is None or empty
    if img is None or not img:
        return []

    exclusion_list = ['2in1']
    recognized_text = []

    # Extract the recognized text from the OCR result
    for line in img:
        # Ensure that line is iterable
        if line is None:
            continue

        for word_info in line:
            # Ensure word_info is in the expected format
            if word_info and len(word_info) > 1 and word_info[1] and len(word_info[1]) > 0:
                recognized_text.append(word_info[1][0])

    # Filter out words that are in the exclusion list


    return recognized_text



"""##**run_with_ocr**"""

import torch
import concurrent.futures

# Function to prepare inputs and run inference with provided OCR text
def run_with_ocr(image, cleaned_text, timeout=60):
    # Update the message to include the cleaned OCR text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # Your image object
                },
                {
                    "type": "text",
                    "text": f"""Can you extract the details from the packaging material given in the image and the provided OCR text?
                    The cleaned OCR text is as follows:
                    "{cleaned_text}"

                    Focus on extracting the following attributes:
                    1. Brand Name
                    2. Pack Size (e.g., weight, volume)
                    3. Product Description

                    The expected format is:
                    Brand Name: {{brand_name if detected else 'N/A'}}
                    Pack Size: {{pack_size if detected else 'N/A'}}
                    Product Description: {{product_description if detected else 'N/A'}}

                    If any attribute cannot be detected, strictly return 'N/A' for that attribute.""",
                },
            ],
        }
    ]

    # Prepare inputs (you may need to adapt this based on your processor and image handling)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    def model_inference():
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

    try:
        # Use concurrent futures for timeout handling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(model_inference)
            output = future.result(timeout=timeout)  # timeout in seconds
            return output

    except concurrent.futures.TimeoutError:
        print(f"Inference timed out after {timeout} seconds.")
        return {"error": "Inference timed out"}
    except Exception as e:
        print(f"Inference failed: {e}")
        return {"error": f"Inference failed: {str(e)}"}

# Example usage
# cleaned_text = "Brand: ExampleBrand, Pack Size: 500g, Description: Example Product, MRP: 100 INR"
# output = run_wout_ocr(image, cleaned_text)
# print(output)

import os
from PIL import Image

# Define the path to your image folder
# image_folder = '/content/drive/MyDrive/#QWN MODEL/Dal_Pulses'
image_folder = '/content/drive/MyDrive/###DL AND ML MODELS /Qwen2-VL-for-OCR-VQA-main-20241005T160243Z-001/atta&flours'
# Loop through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") :  # Add other extensions if needed
        image_path = os.path.join(image_folder, filename)
        # print(f"Processing image: {filename}")

        # Open the image
        image = Image.open(image_path)

        # Pass the image to the run_inference function to extract text
        text = ocr_paddle(image_path)
        print(text)
        extracted_text = run_with_ocr(image,text)

        # Print the extracted text
        print(f"{filename}: {extracted_text}")
        print("------------------------------------")



"""# MRP and Expiry Date Extraction

This script is designed to automatically extract details such as Maximum Retail Price (MRP) and Expiry Date from packaging material images

## Features
The model extracts the following information from an input image of packaging material:
- **MRP (Maximum Retail Price)**: The maximum retail price of the product (in INR).
- **Expiry Date**: The expiration date of the product if available.

If any attribute cannot be detected, the model returns `'N/A'` for that attribute.
"""

import torch
import concurrent.futures

# Function to prepare inputs and run inference with timeout
def Exp_Mrp(image, timeout=60):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # Your image object
                },
                {
                    "type": "text",
                    "text": f"""Can you extract the details from the packaging material given in the image? Specifically, focus on
                    extracting the following attributes:
                    1. MRP (Maximum Retail Price)
                    2. Expiry Date (if available)

                    The expected format is:
                    MRP: {{MRP_value + ' INR' if detected else 'N/A'}}
                    Expiry Date: {{expiry_date if detected else 'N/A'}}

                    If any attribute cannot be detected, strictly return 'N/A' for that attribute.""",
                },
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    def model_inference():
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

    try:
        # Use concurrent futures for timeout handling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(model_inference)
            output = future.result(timeout=timeout)  # timeout in seconds
            return output

    except concurrent.futures.TimeoutError:
        print(f"Inference timed out after {timeout} seconds.")
        return {"error": "Inference timed out"}
    except Exception as e:
        print(f"Inference failed: {e}")
        return {"error": f"Inference failed: {str(e)}"}

import os
from PIL import Image

# Define the path to your image folder
# image_folder = '/content/drive/MyDrive/#QWN MODEL/Dal_Pulses'
image_folder = '/content/'
# Loop through each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") :  # Add other extensions if needed
        image_path = os.path.join(image_folder, filename)
        # print(f"Processing image: {filename}")


        # Open the image
        image = Image.open(image_path)

        # Pass the image to the run_inference function to extract text
        # text = ocr_paddle(image_path)
        # print(text)
        extracted_text = Exp_Mrp(image)

        # Print the extracted text
        print(f"{extracted_text}")
        print("------------------------------------")

"""## Shelf Life Prediction Model

To assess the freshness of produce based on visual cues from an image . The model can generate key information about the shelf life and visual quality of the fresh produce.

## Features
The model predicts the following attributes from an input image:
- **Freshness Level**: Classifies the produce as fresh, ripe, overripe, or spoiled.
- **Predicted Shelf Life**: Estimates the remaining shelf life of the produce (in days).
- **Visual Signs of Deterioration**: Identifies signs like discoloration, bruises, or mold.
- **Overall Quality Score**: Provides a rating of the quality (scale from 1-10).

"""

import torch
import concurrent.futures

# Function to prepare inputs and run inference with timeout
def shelf_life(image, timeout=60):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,  # Your image object
            },
            {
                "type": "text",
                "text": f"""Can you assess the freshness of the fresh produce given in the image by analyzing visual cues? Specifically, focus on extracting the following attributes:
                1. Freshness Level (e.g., fresh, ripe, overripe, spoiled)
                2. Predicted Shelf Life (e.g., in days)
                3. Visual Signs of Deterioration (e.g., discoloration, bruises, mold, etc.)
                4. Overall Quality Score (e.g., a rating from 1-10)

                The expected format is:
                Freshness Level: {{freshness_level }}
                Predicted Shelf Life: {{shelf_life + ' days' }}
                Visual Signs of Deterioration: {{visual_signs }}
                Overall Quality Score: {{quality_score }}""",
            },
        ],
    }
]


    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    def model_inference():
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

    try:
        # Use concurrent futures for timeout handling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(model_inference)
            output = future.result(timeout=timeout)  # timeout in seconds
            return output

    except concurrent.futures.TimeoutError:
        print(f"Inference timed out after {timeout} seconds.")
        return {"error": "Inference timed out"}
    except Exception as e:
        print(f"Inference failed: {e}")
        return {"error": f"Inference failed: {str(e)}"}

import os
from PIL import Image
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/im1.jpg')
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/rapple.jpg')
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/tom.jpeg')
image = Image.open('/content/banana.jpg')
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/mushroom.jpg')
# Pass the image to the run_inference function to extract text
extracted_text = shelf_life(image)

        # Print the extracted text
print(f"{extracted_text}")
print("------------------------------------")

import os
from PIL import Image
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/im1.jpg')
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/rapple.jpg')
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/tom.jpeg')
image = Image.open('/content/banana.jpg')
# image = Image.open('/content/drive/MyDrive/#QWN MODEL/mushroom.jpg')
# Pass the image to the run_inference function to extract text
extracted_text = shelf_life(image)

        # Print the extracted text
print(f"{extracted_text}")
print("------------------------------------")



"""##**LIVE CAPTURING**"""

# Import necessary libraries
import cv2
from google.colab import output
from IPython.display import Javascript, display
import numpy as np
import base64
from PIL import Image
import os
import concurrent.futures
import threading


# Function to decode the base64 image data
def js_to_image(data):
    image_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Function to evaluate JavaScript code
def eval_js(js_code):
    return output.eval_js(js_code)

# Function to take a photo and process it
def take_photo_and_process(quality=10):
    js = Javascript('''
        async function takePhoto(quality) {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Automatically capture after 1 second (1000 ms)
            await new Promise((resolve) => setTimeout(resolve, 1000));

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            // Stop video stream
            stream.getVideoTracks()[0].stop();
            video.remove(); // Remove the video element

            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')

    display(js)

    # Get photo data
    data = eval_js('takePhoto({})'.format(quality))

    # Convert the captured image to OpenCV format
    img = js_to_image(data)

    # Save image directly (optional)
    cv2.imwrite('captured_image.jpg', img)

    # Convert OpenCV image to PIL for processing
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Pass the image to run_wout_ocr function
    def process_t1():
        global t1
        t1 = run_wout_ocr(pil_image)

    # Thread function for MRP processing
    def process_t2():
        global t2
        t2 = Exp_Mrp(pil_image)

    # Create threads
    thread1 = threading.Thread(target=process_t1)
    thread2 = threading.Thread(target=process_t2)

    # Start threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    # Print the extracted text
    print("Extracted Text:", t1+"\n"+t2)


# Call the function to take a photo and process it
take_photo_and_process()

