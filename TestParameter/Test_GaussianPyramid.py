import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread("2a0.jpg",0)
layer = img.copy()
#gaussian pyramid
gp = [layer]

for i in range(2):
	layer = cv2.pyrDown(layer)
	gp.append(layer)
	#cv2.imshow(str(i), layer)
'''
layer = gp[2]
cv2.imshow('upper level', layer)
# laplacian pyramid
lp = [layer]
for i in range(2, 0, -1):
	gaussian_extend = cv2.pyrUp(gp[i])
	laplacian = cv2.subtract(gp[i-1], gaussian_extend)
	cv2.imshow(str(i),laplacian)
cv2.imshow('org',img)
'''
gaussian_extend1 = cv2.pyrUp(gp[1])
gaussian_extend2 = cv2.pyrUp(gp[2])
gaussian_extend2 = cv2.pyrUp(gaussian_extend2)
gaussian_extend3 = cv2.pyrDown(gaussian_extend2)
gaussian_extend3 = cv2.pyrUp(gaussian_extend3)
cv2.imshow(str(1),gaussian_extend1)
cv2.imshow(str(2),gaussian_extend2)
cv2.imshow(str(3),gaussian_extend3)
cv2.imshow(str(0),gp[0])

laplacian1 = cv2.subtract(gp[0], gaussian_extend1)
laplacian2 = cv2.subtract(gaussian_extend1, gaussian_extend2)
laplacian3 = cv2.subtract(gaussian_extend2, gaussian_extend3)
cv2.imshow('laplacian1',laplacian1)
cv2.imshow('laplacian2',laplacian2)
cv2.imshow('laplacian3',laplacian3)
"""
titles = ['image','lr']
images = [img,lr]



for i in range(2):
	plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])
plt.show()
"""
cv2.waitKey(0)
cv2.destroyAllWindows()
