import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os
import sys

def optimize_image(img_path, output_path):
    """
    Enhances an image through resizing, grayscaling, noise reduction,
    thresholding, filtering, and contrast adjustment.

    Parameters:
    - img_path: Path to the input image.
    - output_path: Path to save the optimized image.
    """
    try:
        print("Loading image...")
        img = cv2.imread(img_path)
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

    except Exception as e:
        print(f"An error occurred during image optimization: {e}")

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

if __name__ == "__main__":
    main()
