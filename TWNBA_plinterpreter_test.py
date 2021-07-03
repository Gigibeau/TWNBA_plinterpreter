from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

img = np.array(Image.open("test.tif"))
img2 = Image.open("test.tif")
point1 = np.argmax(img[100,:] > 1000)
point2 = np.argmax(img[900,:] > 1000)
dif= abs(point1-point2)
angle = np.arctan(dif/800)
degree = angle * (180/np.pi)

if point1 > point2:
    rotate_img = img2.rotate(degree)
else:
    rotate_img = img2.rotate(-degree)

rotate_img.save('rotate.tiff')
print(point1)
print(point2)
print(dif)
print(angle)
print(degree)
print(img[1 , 0])