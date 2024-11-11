import boto3
import cv2
import os

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

if __name__ == '__main__':
    image_path = 'image1.jpg'

    bounding_boxes = analyze_document_and_get_bounding_boxes(image_path)

    cropped_image_path = crop_image_to_text(image_path, bounding_boxes)

    extracted_text = extract_text_from_image(cropped_image_path)

    print("Extracted Text from Cropped Image:")
    print(extracted_text)
