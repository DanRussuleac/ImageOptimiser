import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

image = cv2.imread('image.jpg')
resized_image = cv2.resize(image, (450, 600))

cv2.imshow("image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()