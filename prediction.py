# from ultralytics import YOLO
# import cv2

# # Load the trained YOLO model
# model = YOLO('Models/model2(18).pt')  # Replace with your actual model path

# # Perform prediction on the input image
# results = model.predict('output_image_with_boxes.jpg', conf=0.25)

# # Load the original image using OpenCV
# img = cv2.imread('output_image_with_boxes.jpg')

# # Retrieve class names from the model
# class_names = model.names  # Dictionary: {0: 'class0', 1: 'class1', ...}

# # The results object contains the boxes, class labels, and confidence scores
# for result in results:
#     # Extract bounding boxes, class labels, and confidence scores
#     boxes = result.boxes.xyxy  # Bounding box coordinates in (x_min, y_min, x_max, y_max) format
#     confidences = result.boxes.conf  # Confidence scores
#     class_ids = result.boxes.cls  # Class labels (indices)

#     print(len(boxes))
#     # Iterate through each detected object
#     for i in range(len(boxes)):
#         x_min, y_min, x_max, y_max = map(int, boxes[i])  # Convert coordinates to integers
#         confidence = confidences[i].item()  # Convert tensor to float
#         class_id = int(class_ids[i].item())  # Convert tensor to int

#         # Draw bounding box
#         color = (0, 255, 0)  # Green color for the bounding box
#         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

#         # Add a label with the class name and confidence
#         label = f"{class_names[class_id]}: {confidence:.2f}"
#         # # Calculate text size to create a filled rectangle as background for text
#         (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#         cv2.rectangle(img, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
#         cv2.putText(img, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#         # Crop the image using bounding box coordinates
#         cropped_img = img[y_min:y_max, x_min:x_max]

#         # Save or display the cropped image
#         cropped_img_filename = f"cropped_{class_names[class_id]}_{i}.jpg"
#         cv2.imwrite(cropped_img_filename, cropped_img)
#         print(f"Saved cropped image: {cropped_img_filename}")

# # Display the image with bounding boxes (Note: This may not work in some environments like Jupyter notebooks)
# cv2.imshow('Image with Bounding Boxes', img)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import os

# Define the directory to save cropped images
output_dir = "cropped_images"

# Create the directory if it doesn't already exist
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    except OSError as e:
        print(f"Error creating directory '{output_dir}': {e}")
        exit()

# Initialize the YOLO model
model = YOLO('Models/best-train.pt')  # Replace with your actual model path

# Retrieve class names from the model
class_names = model.names  # Dictionary: {0: 'class0', 1: 'class1', ...}

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Store previous objects to compare if they change
previous_objects = []
cropped_images = []
image_id = 0  # Counter for image IDs

def detect_objects(frame):
    """Detect objects and return detected objects info and cropped images."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, conf=0.1, verbose=False)

    current_objects = []
    cropped_images = []

    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class labels (indices)

        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = map(int, boxes[i])
            confidence = confidences[i].item()
            class_id = int(class_ids[i].item())

            # Record the detected object info
            object_info = (class_id, confidence, x_min, y_min, x_max, y_max)
            current_objects.append(object_info)

            # Crop the image using bounding box coordinates
            cropped_img = frame[y_min:y_max, x_min:x_max]
            cropped_images.append((cropped_img, class_id))

    return current_objects, cropped_images

print("Starting real-time object detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect objects on the current frame
    current_objects, new_cropped_images = detect_objects(frame)

    # Check if the objects are different from the previous capture
    if current_objects != previous_objects:
        print("Objects have changed, updating cropped images.")
        previous_objects = current_objects
        cropped_images = new_cropped_images

    # Display the frame with bounding boxes
    for obj in previous_objects:
        class_id, confidence, x_min, y_min, x_max, y_max = obj
        label = f"{class_names[class_id]}: {confidence:.2f}"
        color = (0, 255, 0)
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        # (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x_min, y_min), 
                      (x_min , y_min), color, -1)
        # cv2.putText(frame, label, (x_min, y_min), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Real-Time Object Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting real-time object detection.")
        break

# Save the final cropped images when quitting
for cropped_img, class_id in cropped_images:
    image_id += 1
    cropped_img_filename = os.path.join(output_dir, f"cropped_{class_names[class_id]}_{image_id}.jpg")
    cv2.imwrite(cropped_img_filename, cropped_img)
    print(f"Saved cropped image: {cropped_img_filename}")

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Finished processing. Final cropped images are saved in the 'cropped_images' directory.")
