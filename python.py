import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

image = cv2.imread('image.jpg')

cv2.imshow("image", image)
cv2.waitKey(0)