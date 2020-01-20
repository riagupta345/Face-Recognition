# Recognize faces using classification algorithm like KNN, logistic regression, SVM, etc

# 1. Load training data (numpy arrays of all the persons)
	# x values are stored in the numpy arrays
	# y values we have to assign for each person
# 2. Read a video stream using openCV 
# 3. Extract faces out of it (testing data)
# 4. Use KNN to find the prediction of face (int)
# 5. Map predicted ID to name of user
# 6. Display prediction on screen => Bounding box + name

import numpy as np
import cv2
import os

# KNN Code
def distance(v1, v2):
	return np.sqrt(np.sum((v1-v2)**2))

def knn(train, test, k=5):
	m = train.shape[0]
	dist = []

	for i in range(m):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]

		# Compute distance from test point
		d = distance(test, ix)
		dist.append((d, iy))

	# Sort based on distance and get first k entries
	dk = sorted(dist, key = lambda x: x[0]) [:k]

	# Retrieve only the labels
	labels = np.array(dk)[:, -1]

	# Get frequencies of labels
	freq = np.unique(labels, return_counts = True)

	# Find max frequency and corresponding label
	index = np.argmax(freq[1])
	return freq[0][index]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './FaceData/'

face_data = []
labels = []

class_id = 0    # Class label for current given face
names = {}      # Mapping between class_id and file name (ie, face name)

# Data Preparation
for fx in os.listdir(dataset_path):             # List of all files in this path
	if fx.endswith('.npy'):

		names[class_id] = fx[ : -4]             # Create mapping between class id and face name
		print("Loaded file:", dataset_path+fx)

		currFaces = np.load(dataset_path+fx)    # Load the npy file at location fx as an np array
		face_data.append(currFaces)

		# Create class labels for each face image in this file => eg. 7 faces of Ria, so target value will be 0 for each of the 7 faces of Ria, target=1 for each of the 10 faces of user2
		target = class_id * np.ones((currFaces.shape[0],))    # Vector of length = number of faces of this user
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1, 1))

# print(face_dataset.shape)
# print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)  # Concat X and Y matrices (features and target) to send to KNN algo where they will be sliced accordingly
# print(trainset.shape)


# Testing

while True:

	ret, frame = cap.read()
	if ret == False:
		continue
	frame = cv2.resize(frame, (640, 360))

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)

	for face in faces:
		x, y, w, h = face

		# Get face region of interest
		offset = 10
		face_section = frame[y-offset : y+h+offset , x-offset : x+w+offset]
		face_section = cv2.resize(face_section, (100,100))

		# Predicted label for this face section 
		output = knn(trainset, face_section.flatten())

		# Display predicted name and rectangle
		pred_name = names[int(output)]
		textSize = cv2.getTextSize(pred_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
		
		cv2.putText(frame, pred_name, (x + w//2 - textSize//2, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

	cv2.imshow("Video", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()