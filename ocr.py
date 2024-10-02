import cv2
import pytesseract

# Specify the full path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image):
    """
    Extract text from the original image using Tesseract OCR without preprocessing.
    """
    config = '--psm 6'  # Assume a single uniform block of text
    text = pytesseract.image_to_string(image, config=config)
    return text

def main():
    # Path to the image
    image_path = 'image2.jpg'  # Update this path to your image file
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image at path: {image_path}")
        return

    # Extract text from the original image (no preprocessing)
    extracted_text = extract_text(image)
    
    print("Extracted Text:")
    print(extracted_text)

if __name__ == "__main__":
    main()
