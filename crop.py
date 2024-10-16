import cv2
import numpy as np

def preprocess_image(image):
    """
    Convert the image to grayscale and apply GaussianBlur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

def detect_edges(image):
    """
    Detect edges using the Canny edge detector.
    """
    edges = cv2.Canny(image, 50, 150)
    
    return edges

def apply_morphological_transformations(edges):
    """
    Apply dilation and closing to merge text lines into large text blocks.
    """
    # Define a rectangular kernel for dilation and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    
    # Dilation to connect edges
    dilated = cv2.dilate(edges, kernel, iterations=3)
    
    # Morphological closing to fill gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    return closed

def find_largest_text_block(closed_image):
    """
    Find the largest contour, which is assumed to contain the largest text region.
    """
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return x, y, w, h

def crop_text_region(image, x, y, w, h):
    """
    Crop the largest text region from the original image.
    """
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

if __name__ == "__main__":
    # Load the input image
    input_image_path = 'image4.jpg'  # Replace with your image path
    image = cv2.imread(input_image_path)
    
    # Preprocess the image (grayscale and blur)
    preprocessed_image = preprocess_image(image)
    
    # Detect edges using Canny edge detector
    edges = detect_edges(preprocessed_image)
    
    # Apply morphological transformations (dilate and close)
    closed_image = apply_morphological_transformations(edges)
    
    # Find the largest text block
    text_region = find_largest_text_block(closed_image)
    
    if text_region:
        x, y, w, h = text_region
        
        # Crop and save the largest text region
        cropped_image = crop_text_region(image, x, y, w, h)
        #cv2.imwrite('cropped_text_region.jpg', cropped_image)
        
        # Display the cropped image
        cv2.imshow('Cropped Text Region', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No text region found.")
