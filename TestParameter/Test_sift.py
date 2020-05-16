import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

retval = cv2.getGaussianKernel(5,1.6,cv2.CV_32F)
print("Gaussian kernel----")
print(retval)
print("fuckyouaskdaskd")
#img1 = cv2.imread('2a0.jpg')  # queryImage

#img1 = cv2.imread('2b0.jpg')  # queryImage
img1 = cv2.imread('2b0-1.jpg')  # queryImage
img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#img_set = [[img1a,"xe_do"],[img1b,"ma_do"],[img1c,"si_do"]]

# Initiate SIFT detector

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1 = sift.detect(img2,None)
print('--Hehehe--')
count = 0
for kp in kp1:
    count += 1
    print(kp.response)
print(count)
print('--finished--')
img1 = cv2.drawKeypoints(img2,kp1, img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
#cv2.imwrite('good4.jpg', img1)
cv2.imshow('good4',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()