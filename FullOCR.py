import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import boto3
import os
import sys

# ----------------------------
# Cropping Functions
# ----------------------------

def preprocess_image(image):
    """
    Convert the image to grayscale and apply GaussianBlur.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closed

def find_largest_text_block(closed_image):
    """
    Find the largest contour, which is assumed to contain the largest text region.
    """
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h

def crop_text_region(image, x, y, w, h):
    """
    Crop the largest text region from the original image.
    """
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# ----------------------------------
# Optimization Function with Shadow Removal
# ----------------------------------

def optimize_image(img):
    try:
        print("Removing shadows...")
        rgb_planes = cv2.split(img)

        result_planes = []
        result_norm_planes = []

        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        
        result = cv2.merge(result_planes)
        img = result

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

        print("Image optimization complete!")
        return img_enhanced

    except Exception as e:
        print(f"An error occurred during image optimization: {e}")
        return None

# -----------------------------
# Preprocessing for Textract
# -----------------------------

def preprocess_for_textract(image):
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        print("Converting image to JPEG format for Textract...")
        is_success, buffer = cv2.imencode(".jpg", image_cv)
        if not is_success:
            print("Error encoding image to JPEG format.")
            return None

        image_bytes = buffer.tobytes()

        if len(image_bytes) > 5 * 1024 * 1024:
            print("Image size exceeds 5 MB limit. Resizing image...")
            scale_factor = 0.9 
            while len(image_bytes) > 5 * 1024 * 1024 and scale_factor > 0:
                new_width = int(image_cv.shape[1] * scale_factor)
                new_height = int(image_cv.shape[0] * scale_factor)
                resized_image = cv2.resize(image_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
                is_success, buffer = cv2.imencode(".jpg", resized_image)
                if not is_success:
                    print("Error encoding resized image to JPEG format.")
                    return None
                image_bytes = buffer.tobytes()
                scale_factor -= 0.1 
                image_cv = resized_image

            if len(image_bytes) > 5 * 1024 * 1024:
                print("Unable to reduce image size below 5 MB limit.")
                return None

        print("Image preprocessing for Textract complete.")
        return image_bytes

    except Exception as e:
        print(f"An error occurred during preprocessing for Textract: {e}")
        return None

# -----------------------------
# OCR Function
# -----------------------------

def extract_text_from_image(image_bytes):
    textract = boto3.client('textract')

    try:
        response = textract.detect_document_text(Document={'Bytes': image_bytes})
    except Exception as e:
        print("An error occurred while calling Textract:")
        print(e)
        return ""

    detected_text = '\n'.join(
        block['Text']
        for block in response.get('Blocks', [])
        if block['BlockType'] == 'LINE'
    )
    return detected_text

# -----------------------------
# Main Execution Block
# -----------------------------

if __name__ == "__main__":
    try:
        src_path = os.path.dirname(os.path.abspath(__file__))

        input_image = "image6.jpg" 
        img_path = os.path.join(src_path, input_image)

        if not os.path.isfile(img_path):
            print(f"Error: The image file '{input_image}' does not exist in '{src_path}'.")
            sys.exit(1)

        print("Loading the input image...")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Unable to load image at {img_path}.")
            sys.exit(1)

        print("Cropping the largest text region...")
        preprocessed_image = preprocess_image(image)
        edges = detect_edges(preprocessed_image)
        closed_image = apply_morphological_transformations(edges)
        text_region = find_largest_text_block(closed_image)

        if text_region:
            x, y, w, h = text_region
            cropped_image = crop_text_region(image, x, y, w, h)
            cv2.imwrite('cropped_text_region.jpg', cropped_image)
        else:
            print("No text region found.")
            sys.exit(1)

        print("Optimizing the cropped image...")
        optimized_image = optimize_image(cropped_image)
        if optimized_image is None:
            print("Image optimization failed.")
            sys.exit(1)

        processed_image_path = os.path.join(src_path, 'processed_image.png')
        optimized_image.save(processed_image_path)
        print(f"Processed image saved as '{processed_image_path}'.")

        print("Preprocessing optimized image for Textract...")
        image_bytes = preprocess_for_textract(optimized_image)
        if image_bytes is None:
            print("Preprocessing for Textract failed.")
            sys.exit(1)

        print("Extracting text from optimized image...")
        extracted_text = extract_text_from_image(image_bytes)

        print("Extracted Text from Optimized Image:")
        print(extracted_text)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
