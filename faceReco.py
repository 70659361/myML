import cv2 as cv

FILE_PATH="./dataset/face/"

p=cv.imread(FILE_PATH+"jelly/326676628.jpg",1)
cv.imshow("img", p)
cv.waitKey(0)