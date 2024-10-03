import boto3
import cv2
import os

def analyze_document_and_get_bounding_boxes(image_path):
    # Initialize Amazon Textract client
    textract = boto3.client('textract')

    # Open the image file
    with open(image_path, 'rb') as document:
        image_bytes = bytearray(document.read())

    # Call Textract to detect text and get bounding boxes
    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    bounding_boxes = []
    # Iterate through detected blocks and capture bounding boxes for lines of text
    for block in response['Blocks']:
        if block['BlockType'] == 'LINE':
            bbox = block['Geometry']['BoundingBox']
            bounding_boxes.append(bbox)
    
    return bounding_boxes

def crop_image_to_text(image_path, bounding_boxes):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Get image dimensions

    # Initialize minimum and maximum coordinates for the bounding rectangle
    min_left, min_top = w, h
    max_right, max_bottom = 0, 0

    # Convert bounding box coordinates from relative to actual image coordinates
    for bbox in bounding_boxes:
        left = int(bbox['Left'] * w)
        top = int(bbox['Top'] * h)
        width = int(bbox['Width'] * w)
        height = int(bbox['Height'] * h)
        
        # Update the minimum and maximum coordinates to find the enclosing rectangle
        min_left = min(min_left, left)
        min_top = min(min_top, top)
        max_right = max(max_right, left + width)
        max_bottom = max(max_bottom, top + height)

    # Crop the region containing all text
    cropped_image = image[min_top:max_bottom, min_left:max_right]

    # Save the cropped image
    cropped_image_path = os.path.join(os.getcwd(), 'cropped_text_image.jpg')
    cv2.imwrite(cropped_image_path, cropped_image)
    print(f"Cropped image saved at: {cropped_image_path}")

    return cropped_image_path

def extract_text_from_image(image_path):
    # Initialize Amazon Textract client
    textract = boto3.client('textract')

    # Open the cropped image file
    with open(image_path, 'rb') as document:
        image_bytes = bytearray(document.read())

    # Call Textract to detect text in the cropped image
    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    # Extract the detected text
    detected_text = ''
    for block in response['Blocks']:
        if block['BlockType'] == 'LINE':
            detected_text += block['Text'] + '\n'

    return detected_text

# Path to the original image
image_path = 'image4.jpg'

# Get bounding boxes from Textract
bounding_boxes = analyze_document_and_get_bounding_boxes(image_path)

# Crop the image to a single region containing all the text
cropped_image_path = crop_image_to_text(image_path, bounding_boxes)

# Run OCR on the cropped image
extracted_text = extract_text_from_image(cropped_image_path)

# Output the extracted text
print("Extracted Text from Cropped Image:")
print(extracted_text)
