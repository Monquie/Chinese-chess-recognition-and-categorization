import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.8
fontColor              = (255,255,0)
lineType               = 2

radius = 3  
color = (0, 0, 255) 
thickness = 2
number = 0
# caliberation parameter
fu = 642.0662
fv = 642.0836
fux = -0.3454
uc = 322.6147
vc = 244.645

R = [[0, -0.9995, 0.0292],
	[0.9986, 0.0115, 0.0518],
	[-0.0521, 0.0287, 0.9982]]
R = np.linalg.inv(R)
D = [[-31.843],
	[158.0205],
	[533.3379]]

print(D[2][0])
# cot = (-fux/fu)
# cosphi = np.absolute(sqrt(cot**2/(cot**2+1)))
# sinphi = np.absolute(sqrt(1-cosphi**2))
# (D[2][0]*(circle[1]-vc))*(-fux/fu)/fv
def tranform(circle):
	X = 85 + (D[2][0]*(circle[0]-uc))/fu #sua lai he so 85 cho phu hop +85
	Y = (D[2][0]*(circle[1]-vc))/fv
	Pcam = [[X],[Y],[D[2][0]]]
	Temp = np.subtract(Pcam, D)
	Point2World = np.dot(R, Temp)
	return Point2World;

print(tranform([522,426]))


cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('o'):
		start_time = time.time()
		xmg = frame.copy()
		img = cv2.cvtColor(xmg, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (5, 5), 0)
		cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
		h, w = img.shape

		circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30, param1=220,param2=40,minRadius=30,maxRadius=35) #min 30, max 40
		#print("--- %s seconds ---" % (time.time() - start_time))
		if circles is None:
			continue
		else:
			circles = circles[0][0]
			print(circles[0])
			print(circles[1])
			print('ban kinh duong tron la: ')
			print(circles[2])
			WorldPoint = tranform(circles)
			circles = np.uint16(np.around(circles))
			print(WorldPoint)

			img = cv2.circle(xmg,(circles[0], circles[1]), radius, color, thickness)
			print("time cost "+ str(time.time()-start_time))

			#print time_for one process	
			cv2.imshow("img",img)
			#cv2.imwrite('testla{}.jpg'.format(number),xmg)

	elif cv2.waitKey(1) & 0xFF == 27:
		break
cap.release()
cv2.destroyAllWindows()
