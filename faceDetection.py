import cv2

cap = cv2.VideoCapture(0)    
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 
# This is a classifier trained on facial data


while True:
	ret, frame = cap.read()     
	if ret == False:            
		continue

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	frame = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)   # Scale factor and number of neighbours
	# This function returns a list of tuples => one tuple for each face detected
	# Each tuple has (x, y, w, h) => (x, y) is starting point of rectangular region and (w, h) are width and height of rectangular region

	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		# Takes coordinates of top left and bottom right points, and color and thickness of border


	cv2.imshow("Video Frame", frame)
	# cv2.imshow("Gray Video Frame", gray_frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()