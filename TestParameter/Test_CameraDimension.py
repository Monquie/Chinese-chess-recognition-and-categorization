import cv2
import datetime
cap = cv2.VideoCapture(0)

#cap.set(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1208)
#cap.set(3, 400) # 3 ~ WIDTH
#cap.set(4, 400)  # 4 ~ HEIGHT

print(cap.get(3))
print(cap.get(4))

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:

		font = cv2.FONT_HERSHEY_SIMPLEX
		text = 'Width:' + str(cap.get(3)) + ' Height:' +str(cap.get(4))
		date = str(datetime.datetime.now()) #date time
		frame = cv2.putText(frame, date, (10,50), font, 1, (0,255,255), 2, cv2.LINE_AA)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame',frame)


		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.imwrite('fuckyoubitch.jpg',frame)
			break

cap.release()
cv2.destroyAllWindows()