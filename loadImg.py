import cv2 
import os
import numpy as np 
from PIL import Image

img = cv2.imread('/home/trung/Documents/Cafee API/Save/01-21_thieu_canxi.jpg')
img = cv2.resize(img, (224, 224))
print(img)