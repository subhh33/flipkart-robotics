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

if __name__ == "__main__":
    main()
