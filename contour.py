import numpy as np
import cv2 as cv
im = cv.imread('res.png')
im = im[0:250,0:190]
points = []
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv.contourArea(contour) > 0:
        cv.drawContours(im, contour, 0, (0,255,0), 3)
        x,y,w,h = cv.boundingRect(contour)
        cv.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        points.append([(x+w)/2,(y+h)/2])
print(points)
cv.imshow('Contours', im)
cv.waitKey(0)