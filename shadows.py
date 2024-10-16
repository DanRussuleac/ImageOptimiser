import cv2
import numpy as np

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
result_norm = cv2.merge(result_norm_planes)

# Display the original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Shadow Removed Image', result)

# Save the processed image
#cv2.imwrite('no_shadow_image.jpg', result)

cv2.waitKey(0)
cv2.destroyAllWindows()