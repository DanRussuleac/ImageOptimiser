import cv2
import numpy as np
from matplotlib import pyplot as plt
import easygui
import pytesseract
import os

# ----------------------------
# Configuration for Tesseract
# ----------------------------

# Specify the full path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ----------------------------
# Function Definitions
# ----------------------------

def resize_image(image, width=450, height=600):
    """
    Resize the image to the specified width and height.
    
    Parameters:
        image (numpy.ndarray): The original image.
        width (int): The desired width in pixels.
        height (int): The desired height in pixels.
        
    Returns:
        resized (numpy.ndarray): The resized image.
    """
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.
    
    Steps:
    - Convert to grayscale
    - Apply noise reduction
    - Apply thresholding
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction with bilateral filter
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text(image):
    """
    Extract text from the preprocessed image using pytesseract.
    """
    # Configuration for Tesseract
    config = '--psm 6'  # Assume a single uniform block of text
    
    # Extract text
    text = pytesseract.image_to_string(image, config=config)
    
    return text

def save_text(text, output_path):
    """
    Save the extracted text to a .txt file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

# ----------------------------
# Main Script
# ----------------------------

def main():
    # Step 1: Open a file dialog to select an image
    image_path = easygui.fileopenbox(title="Select an Image for OCR",
                                    filetypes=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"])
    
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    # Step 2: Read the image using OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image at path: {image_path}")
        return
    
    # Step 3: Resize the image
    desired_width = 450  # You can adjust these values as needed
    desired_height = 600
    resized_image = resize_image(image, width=desired_width, height=desired_height)
    
    # Optional: Display the resized image
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Step 4: Preprocess the resized image for better OCR accuracy
    preprocessed_image = preprocess_image(resized_image)
    
    # Optional: Display the preprocessed image
    # plt.imshow(preprocessed_image, cmap='gray')
    # plt.title('Preprocessed Image')
    # plt.axis('off')
    # plt.show()
    
    # Step 5: Extract text from the preprocessed image
    extracted_text = extract_text(preprocessed_image)
    
    print("Extracted Text:")
    print(extracted_text)
    
    # Step 6: Save the extracted text to a .txt file
    # Define the output path (same directory as image)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.dirname(image_path)
    output_path = os.path.join(output_dir, f"{base_name}_extracted.txt")
    
    save_text(extracted_text, output_path)
    
    print(f"Extracted text saved to: {output_path}")

if __name__ == "__main__":
    main()
