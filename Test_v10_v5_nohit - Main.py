from __future__ import print_function
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

import serial
import datetime
 
from collections import Counter

#=======================================================================
# import the necessary packages
from threading import Thread,Event
import queue

class LocalVideoStream:
	def __init__(self, src):
		# initialize the video camera stream and read the first img
		# from the stream
		self.stream = cv2.VideoCapture(src)
		self.success, self.a = self.stream.read()
		if self.success:
			self.img = self.a

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read imgs from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next img from the stream
			self.success, self.img = self.stream.read()

	def read(self):
		# return the img most recently read
		return self.img

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

img1a = cv2.imread('2a0.jpg',0)  # queryImage
img1b = cv2.imread('2b0.jpg',0)  # queryImage
img1c = cv2.imread('2c0.jpg',0)  # queryImage
img1d = cv2.imread('2d0.jpg',0)  # queryImage
img1e = cv2.imread('2e0.jpg',0)  # queryImage
img1f = cv2.imread('2f0.jpg',0)  # queryImage
img1g = cv2.imread('2g0.jpg',0)  # queryImage
img1h = cv2.imread('2h0.jpg',0)  # queryImage
img1i = cv2.imread('2i0.jpg',0)  # queryImage
img1j = cv2.imread('2j0.jpg',0)  # queryImage
img1k = cv2.imread('2k0.jpg',0)  # queryImage
img1l = cv2.imread('2l0.jpg',0)  # queryImage
img1m = cv2.imread('2m0.jpg',0)  # queryImage
img1n = cv2.imread('2n0.jpg',0)  # queryImage
img_set = [[img1a,"xe_do"],[img1b,"ma_do"],[img1c,"si_do"],[img1d,"vua_do"],[img1e,"tinh_do"],[img1f,"tot_do"],[img1g,"phao_do"],
           [img1h,"tot_den"],[img1i,"ma_den"],[img1j,"vua_den"],[img1k,"xe_den"],[img1l,"si_den"],[img1m,"phao_den"],[img1n,"tinh_den"]]
#  chess_set                       trainImage
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

image_feature = []
for img in img_set:
	_, des1 = sift.detectAndCompute(img[0],None)
	image_feature.append(des1)

# Initiate SIFT detector
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# define crop image function
def ret_vertices(circle,w,h):
	bottomleft = []
	topright = []
	if circle[0] > circle[2]:
		bottomleft = [circle[0]-circle[2],circle[1]+circle[2]]
	else:
		bottomleft = [0,circle[1]+circle[2]]
	if circle[1] > circle[2]:
		topright = [circle[0]+circle[2],circle[1]-circle[2]]
	else:
		topright = [circle[0]+circle[2],0]
	if bottomleft[1] > h:
		bottomleft[1] = h
	if topright[0] > w:
		topright[0] = w
	return [bottomleft,topright,[circle[0],circle[1]]]
def ret_chess_img(rectangle, img_):
	crop_img = img_[rectangle[1][1]:rectangle[0][1],rectangle[0][0]:rectangle[1][0]].copy() #crop_img = img[y:y+h, x:x+w]
	return crop_img

def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]

# define font for results
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1.25
fontColor              = (0,0,255) # Red Results
lineType               = 2

radius = 3  
color = (0, 0, 255) 
thickness = 2
# create a Set of Circle Point
CircleSet	=	None
chess_name	=	None
chess_hit	=	None
Chess_set_global = []
QueueSet	=	queue.Queue()
QueueImage	=	queue.LifoQueue()
Event		=	Event()
# Ser 		=	serial.Serial('COM10', 9600, timeout=1)
Mask		=	np.zeros((480,640,1), np.uint8)
for i in range(640):
    for j in range(480):
        Mask[j,i] = 255;
        if i>480:
            Mask[j,i] = 0;

# created a *threaded *video stream, allow the camera senor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED imgs from webcam...")
vs = LocalVideoStream(src=0).start()
background = vs.read()

def FindCircles():
	global 	CircleSet
	global	QueueImage
	global	QueueSet
	while(vs.success):
		start_time = time.time()
		InputImage = vs.read()
		ProcessImage = cv2.absdiff(background, InputImage)
		img = cv2.cvtColor(ProcessImage, cv2.COLOR_BGR2GRAY)

		img = cv2.GaussianBlur(img, (5, 5), 0)
		img = cv2.bitwise_and(img, Mask)
		# CircleSet_org = cv2.HoughCircles(image=img,method=cv2.HOUGH_GRADIENT,dp=1,minDist=30, 
		# 							param1=25,param2=18,minRadius=30,maxRadius=40) 
		CircleSet_org = cv2.HoughCircles(image=img,method=cv2.HOUGH_GRADIENT,dp=1,minDist=30, 
									param1=220,param2=40,minRadius=27,maxRadius=33) #change param1 = 60

		if CircleSet_org is None:
			cv2.imshow('ImageResult',ProcessImage)
		else:
			CircleSet = np.uint16(np.around(CircleSet_org[0]))
			arr_temp  =	CircleSet[:,0]
			index_max = np.where(arr_temp == np.amax(arr_temp))[0][0]
			CircleSet = CircleSet[index_max]
			if CircleSet[0] >= 460:
				CircleSet	=	None
				if not QueueSet.empty():
					chess_hit		=	QueueSet.get()
					print('QueueSet is empty: ' + str(QueueSet.empty()))

			# Starting the Detector
			if QueueSet.empty():
				if QueueImage.qsize() > 100:
					with QueueImage.mutex:
						QueueImage.queue.clear()
				QueueImage.put(InputImage)
				Event.set()

			if CircleSet is None:
				continue
			else:
				ShowImage	=	cv2.circle(InputImage, (CircleSet[0],CircleSet[1]), radius, color, thickness)
				print('cordinate is ('+ str(CircleSet[0]) + ', ' +str(CircleSet[1])+')')
				if not QueueSet.empty():
					ShowImage	=	cv2.putText(InputImage,chess_name, (10,40), font, 
												fontScale, fontColor, lineType)
				cv2.imshow('ImageResult', ShowImage)
			print("---FindCircle Done in %s seconds ---" % (time.time() - start_time))

		if cv2.waitKey(1) & 0xFF == 27:
			vs.stop()
			break;

def DetectCharacter():
	global chess_name
	global Chess_set_global
	count = 0
	while(vs.success):
		Event.wait()
		start_time = time.time()
		print('starting the detector')
		if CircleSet is None:
			print(CircleSet)
			pass
		else:
			ImageShow	=	QueueImage.get()
			print('size of QueueImage is: '+str(QueueImage.qsize()))
			vertices_set	=	ret_vertices(CircleSet,
											ImageShow.shape[1],ImageShow.shape[0])

			chess_img = ret_chess_img([vertices_set[0],vertices_set[1]],ImageShow)
			if chess_img is None or chess_img.shape[0]!=chess_img.shape[1] or chess_img.shape[0] == 0:
				pass
			else:
				chess_center = vertices_set[2]
				chess_set=([chess_img,chess_center])

				matches_num = []
				print(chess_set[0].shape)
				_, des2 = sift.detectAndCompute(chess_set[0],None)
				for Description in image_feature:
					matches = flann.knnMatch(Description,des2,k=2)
					good = []
					for m,n in matches:
						if m.distance < 0.7*n.distance:
							good.append(m)
					matches_num.append(len(good))
				if max(matches_num) >= 3:
					chess_set.append(img_set[matches_num.index(max(matches_num))][1])
				else:
					chess_set.append('None')
				print("---Done in %s seconds ---" % (time.time() - start_time))
				Chess_set_global.append(chess_set[2])
				count 	+=	1
				if count == 20:
					print(Chess_set_global)
					chess_name	=	most_frequent(Chess_set_global)
					QueueSet.put(most_frequent(Chess_set_global))
					Chess_set_global = []
					count = 0
			# print("---Done in %s seconds ---" % (time.time() - start_time))
		Event.clear()


FindCircleThread = Thread(target	=	FindCircles, daemon = True)
FindCharacter	 = Thread(target	=	DetectCharacter, daemon = True)

FindCircleThread.start()
FindCharacter.start()
