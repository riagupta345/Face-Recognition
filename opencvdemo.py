import cv2

img = cv2.imread('dog.png')
grayImg = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Dog Image', img)
cv2.imshow('Gray Dog Image', grayImg)

cv2.waitKey(0)
# cv2.waitKey(250)
cv2.destroyAllWindows