# Write a Python script that captures images from webcam video stream
# - Extracts all faces from image frame using haarcascades 
# - Stores the face information into numpy arrays

# 1. Read and store video stream, capture images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import numpy as np
import cv2

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './FaceData/'
file_name = input("Enter name of person: ")

while True:

	ret, frame = cap.read()
	if ret == False:
		continue
	
	frame = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	# print(len(faces), type(faces), faces)
	
	# if len(faces) == 0:
	# 	continue

	# If face is found: type is numpy ndarray of shape (k, 4) <k is no. of faces> , where each face is [x, y, w, h] 
	# If face is not found: type is empty tuple
		
	if len(faces) != 0:
		faces = sorted(faces, key = lambda f: f[2]*f[3])      # Sort on the basis of area, ie, width * height (3rd and 4th parameter of each list entry)
		# print(faces)

		# Pick only the last face because it is the largest face according to area
		# for face in faces[-1:]: 
		face = faces[-1]
		x, y, w, h = face
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

		# Extract (crop out) the required face: Region of Interest
		offset = 10  # For padding of 10 on all sides (TBRL)
		face_section = frame[y-offset : y+h+offset , x-offset : x+w+offset]         # In frame, rows for Y-axis, columns for X-axis
		# print(face_section.size)
		if face_section.size == 0:
			continue
		face_section = cv2.resize(face_section, (100,100))

		if skip%10 == 0:
			face_data.append(face_section)
			# print(len(face), len(face_data))

		skip += 1
		cv2.imshow("Face Section", face_section)

	cv2.imshow("Video", frame)
	

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert face list array into a numpy array
face_data = np.asarray(face_data)
# print(face_data.shape) # => numOfFacesCapturedInAboveLoop * 100*100*3 (100x100 image and 3 channels?)

face_data = face_data.reshape((face_data.shape[0], -1))       # Number of rows should be num of faces, and 100*100*3 image should be changed into a single row
# print(face_data.shape) 

# Save data into the file system
np.save(dataset_path+file_name+".npy", face_data)
print("Data saved at", dataset_path+file_name+".npy")
cap.release()
cv2.destroyAllWindows()