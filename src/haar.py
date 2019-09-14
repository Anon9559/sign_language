
import numpy as np
import cv2 as cv


img = cv.imread('../tests/images/hand.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.equalizeHist(gray)

# Detect hands using classifier.
hand_cascade = cv.CascadeClassifier('../data/aGest.xml')
hands = hand_cascade.detectMultiScale(img)

# Draw bounding boxes round detected hands.
for (x, y, w, h) in hands:
    img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w] 
    roi_img = img[y:y + h, x:x + w]


cv.imshow('output', img)

cv.waitKey(0)
cv.destroyAllWindows()

