import cv2

cap = cv2.VideoCapture(0)       # Takes input as webcamID => webcamID = 0 for default webcam, specify if there are many webcams

print(cap.get(3), cap.get(4))


while True:
	ret, frame = cap.read()     # Returns 2 parameters, ret: if frame was successfully read, and the image frame

	if ret == False:            # If webcam has not started, or any other error
		continue

	frame = cv2.resize(frame, (640,360), interpolation=cv2.INTER_AREA)
	cv2.imshow("Video Frame", frame)

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("Gray Video Frame", gray_frame)

	key_pressed = cv2.waitKey(1) & 0xFF # Take keyboard input every 1ms, so waitKey returns a 32 bit string, which is anded with 8 ones to get last 8 bits of the key pressed
	if key_pressed == ord('q'):         # ord('q') returns ASCII value of char q
		break


cap.release()
cv2.destroyAllWindows()