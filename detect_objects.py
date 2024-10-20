from ultralytics import YOLO
import cv2
import os
import json
from flipkart_robotics import take_photo_and_process  # Import the required function

# Define the directory to save cropped images
output_dir = "cropped_images"

# Create the directory if it doesn't already exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the YOLO model
model = YOLO('Models/best-train.pt')  # Path to your YOLO model

# Retrieve class names from the model
class_names = model.names  # Dictionary of class names

def process_image(frame):
    """Detect objects in the given frame and return the count and details."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, conf=0.25, verbose=False)

    object_count = 0
    object_details = []

    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class labels (indices)

        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = map(int, boxes[i])
            confidence = confidences[i].item()
            class_id = int(class_ids[i].item())

            # Record the detected object info
            object_info = {
                "class_name": class_names[class_id],
                "confidence": confidence,
                "bounding_box": [x_min, y_min, x_max, y_max]
            }
            object_details.append(object_info)
            object_count += 1

            # Crop the image using bounding box coordinates
            cropped_img = frame[y_min:y_max, x_min:x_max]
            cropped_img_filename = os.path.join(output_dir, f"cropped_{class_names[class_id]}_{object_count}.jpg")
            cv2.imwrite(cropped_img_filename, cropped_img)
            print(f"Saved cropped image: {cropped_img_filename}")

            # Call the function to process the cropped image and get details
            details = take_photo_and_process(cropped_img_filename)
            object_details[-1]["additional_details"] = details  # Store additional details from processing

    return object_count, json.dumps(object_details, indent=4)  # Return count and details as JSON
