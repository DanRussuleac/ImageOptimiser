import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os
import sys

def optimize_image(img_path, output_path):
    """
    Optimizes an image by resizing, converting to grayscale, applying dilation and erosion,
    adaptive thresholding, median filtering, and contrast enhancement.

    Parameters:
    - img_path: Path to the input image.
    - output_path: Path to save the optimized image.
    """
    try:
        print("✅ Loading image...")
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Error: Unable to load image at {img_path}.")
            return
        
        print("🔄 Resizing image by a factor of 3...")
        img_resized = cv2.resize(img, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        
        print("🖤 Converting image to grayscale...")
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        print("🔍 Applying dilation and erosion to reduce noise...")
        kernel = np.ones((3, 3), np.uint8)  # Increased kernel size for better noise reduction
        img_dilated = cv2.dilate(img_gray, kernel, iterations=1)
        img_eroded = cv2.erode(img_dilated, kernel, iterations=1)
        
        print("⚙️ Applying adaptive thresholding...")
        img_thresh = cv2.adaptiveThreshold(
            img_eroded,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Increased block size for better local thresholding
            5    # Increased constant for optimal separation
        )
        
        print("🧹 Applying median filter to remove residual noise...")
        img_pil = Image.fromarray(img_thresh)
        img_filtered = img_pil.filter(ImageFilter.MedianFilter(size=3))  # Specified filter size
        
        print("🔗 Enhancing image contrast...")
        enhancer = ImageEnhance.Contrast(img_filtered)
        img_enhanced = enhancer.enhance(2)  # Adjust the factor as needed
        
        print(f"💾 Saving optimized image as '{output_path}'...")
        img_enhanced.save(output_path)
        
        print("🎉 Image optimization complete!")

    except Exception as e:
        print(f"❌ An error occurred during image optimization: {e}")

def main():
    """
    Main function to execute the image optimization.
    """
    try:
        # Get the directory where the script is located
        src_path = os.path.dirname(os.path.abspath(__file__))
        
        # Define the input and output image filenames
        input_image = "image7.jpg"  # Ensure this file exists in the same folder
        optimized_image = "optimized_image7.png"  # Final optimized image
        
        # Construct the full image paths
        img_path = os.path.join(src_path, input_image)
        output_path = os.path.join(src_path, optimized_image)
        
        # Check if the input image exists
        if not os.path.isfile(img_path):
            print(f"❌ Error: The image file '{input_image}' does not exist in '{src_path}'.")
            sys.exit(1)
        
        # Optimize the image
        optimize_image(img_path, output_path)
    
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
