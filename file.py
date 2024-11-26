import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os
import sys
import boto3

def main():
    """
    Executes the image optimization process.
    """
    try:
        src_path = os.path.dirname(os.path.abspath(__file__))

        input_image = "image7.jpg"
        optimized_image = "optimized_image7.png"

        img_path = os.path.join(src_path, input_image)
        output_path = os.path.join(src_path, optimized_image)

        if not os.path.isfile(img_path):
            print(f"Error: The image file '{input_image}' does not exist in '{src_path}'.")
            sys.exit(1)

        optimize_image(img_path, output_path)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def optimize_image(img_path, output_path):
    # Read the image
    image = cv2.imread('image7.jpg')

    # Split the image into its RGB planes
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []

    # Process each plane separately
    for plane in rgb_planes:
        # Dilate the image to remove small shadows
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        
        # Apply median blur to smoothen the dilated image
        bg_img = cv2.medianBlur(dilated_img, 21)
        
        # Compute the difference between the original plane and the background image
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        
        # Normalize the difference image
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    # Merge the processed planes back into an RGB image
    result = cv2.merge(result_planes)
    """
    Enhances an image through resizing, grayscaling, noise reduction,
    thresholding, filtering, and contrast adjustment.

    Parameters:
    - img_path: Path to the input image.
    - output_path: Path to save the optimized image.
    """
    try:
        print("Loading image...")
        img = result
        if img is None:
            print(f"Error: Unable to load image at {img_path}.")
            return

        print("Resizing image...")
        img_resized = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

        print("Converting to grayscale...")
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        print("Reducing noise...")
        kernel = np.ones((3, 3), np.uint8)
        img_dilated = cv2.dilate(img_gray, kernel, iterations=1)
        img_eroded = cv2.erode(img_dilated, kernel, iterations=1)

        print("Applying adaptive thresholding...")
        img_thresh = cv2.adaptiveThreshold(
            img_eroded,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5
        )

        print("Applying median filter...")
        img_pil = Image.fromarray(img_thresh)
        img_filtered = img_pil.filter(ImageFilter.MedianFilter(size=3))

        print("Enhancing contrast...")
        enhancer = ImageEnhance.Contrast(img_filtered)
        img_enhanced = enhancer.enhance(2)

        print(f"Saving optimized image as '{output_path}'...")
        img_enhanced.save(output_path)

        print("Image optimization complete!")

        crop(img_enhanced)

    except Exception as e:
        print(f"An error occurred during image optimization: {e}")

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

def crop(import_image):
    image = import_image
    
    # Preprocess the image (grayscale and blur)
    #preprocessed_image = preprocess_image(image)
    
    # Detect edges using Canny edge detector
    edges = detect_edges(image)
    
    # Apply morphological transformations (dilate and close)
    closed_image = apply_morphological_transformations(edges)
    
    # Find the largest text block
    text_region = find_largest_text_block(closed_image)
    
    if text_region:
        x, y, w, h = text_region
        
        # Crop and save the largest text region
        cropped_image = crop_text_region(image, x, y, w, h)
        #cv2.imwrite('cropped_text_region.jpg', cropped_image)

        doOCR(cropped_image)
    else:
        print("No text region found.")



def analyze_document_and_get_bounding_boxes(image_path):
    """
    Uses Amazon Textract to detect text and retrieve bounding boxes from an image.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - bounding_boxes (list): List of bounding boxes for each detected line of text.
    """
    textract = boto3.client('textract')

    with open(image_path, 'rb') as document:
        image_bytes = bytearray(document.read())

    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    bounding_boxes = [
        block['Geometry']['BoundingBox']
        for block in response['Blocks']
        if block['BlockType'] == 'LINE'
    ]
    return bounding_boxes

def crop_image_to_text(image_path, bounding_boxes):
    """
    Crops the image to the region containing all detected text.

    Parameters:
    - image_path (str): Path to the original image.
    - bounding_boxes (list): List of bounding boxes for text regions.

    Returns:
    - cropped_image_path (str): Path to the saved cropped image.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Initialize coordinates for the bounding rectangle
    min_left, min_top = w, h
    max_right, max_bottom = 0, 0

    # Convert relative bounding box coordinates to absolute values
    for bbox in bounding_boxes:
        left = int(bbox['Left'] * w)
        top = int(bbox['Top'] * h)
        width = int(bbox['Width'] * w)
        height = int(bbox['Height'] * h)
        
        # Update the bounding rectangle coordinates
        min_left = min(min_left, left)
        min_top = min(min_top, top)
        max_right = max(max_right, left + width)
        max_bottom = max(max_bottom, top + height)

    # Crop and save the image
    cropped_image = image[min_top:max_bottom, min_left:max_right]
    cropped_image_path = os.path.join(os.getcwd(), 'cropped_text_image.jpg')
    cv2.imwrite(cropped_image_path, cropped_image)
    print(f"Cropped image saved at: {cropped_image_path}")

    return cropped_image_path

def extract_text_from_image(image_path):
    """
    Extracts text from an image using Amazon Textract.

    Parameters:
    - image_path (str): Path to the image.

    Returns:
    - detected_text (str): Extracted text content.
    """
    textract = boto3.client('textract')

    with open(image_path, 'rb') as document:
        image_bytes = bytearray(document.read())

    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    detected_text = '\n'.join(
        block['Text']
        for block in response['Blocks']
        if block['BlockType'] == 'LINE'
    )
    return detected_text


def doOCR(import_path):
    image_path = import_path

    # Obtain bounding boxes from Textract
    bounding_boxes = analyze_document_and_get_bounding_boxes(image_path)

    # Crop the image to the region containing text
    cropped_image_path = crop_image_to_text(image_path, bounding_boxes)

    # Extract text from the cropped image
    extracted_text = extract_text_from_image(cropped_image_path)

    # Output the extracted text
    print("Extracted Text from Cropped Image:")
    print(extracted_text)

if __name__ == "__main__":
    main()