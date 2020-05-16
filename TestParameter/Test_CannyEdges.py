import cv2
import numpy as np 
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    ret, img  = cap.read()
    canny = cv2.Canny(img, 110,220)

    cv2.imshow('canny test',canny)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()