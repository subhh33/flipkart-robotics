from ultralytics import YOLO
import cv2
import time

# Initialize the YOLO model
model = YOLO('Models/model2(18).pt')  # Replace with your actual model path

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

def detect_objects(frame):
    """Detect objects and return detected objects info and cropped images."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, conf=0.25, verbose=False)

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
            cropped_images.append(cropped_img)

    return current_objects, cropped_images, len(cropped_images)

print("Starting real-time object detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect objects on the current frame
    current_objects, new_cropped_images, No_of_items = detect_objects(frame)

    # if No_of_items >= 2:
    #     print(f"Number of cropped images detected: {No_of_items}")
    #     cropped_images.extend(new_cropped_images)  # Add newly cropped images to the list
    #     break  # Exit the loop after detecting objects

    # Display the frame with bounding boxes
    for obj in current_objects:
        class_id, confidence, x_min, y_min, x_max, y_max = obj
        label = f"{class_names[class_id]}: {confidence:.2f}"
        color = (0, 255, 0)
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x_min, y_min - text_height - baseline), 
                      (x_min + text_width, y_min), color, -1)
        cv2.putText(frame, label, (x_min, y_min - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Real-Time Object Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting real-time object detection.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Now display the cropped images
if len(cropped_images) > 0:
    for i, cropped_image in enumerate(cropped_images):
        cv2.imshow(f'Cropped Image {i + 1}', cropped_image)
        cv2.waitKey(0)  # Wait for a key press to close the window

cv2.destroyAllWindows()
print("Finished processing. All cropped images have been displayed.")
