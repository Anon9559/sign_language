"""
Only works on single impage inputs
"""
import numpy as np
import cv2
# Load haar classifier (empty tuple).
# N.B. very picky about file path, hence full path.
hand_cascade = cv2.CascadeClassifier('C:/Users/arjun/Documents/git/sign_language/data/aGest.xml')

# Image file path.
img = cv2.imread('C:/Users/arjun/Documents/git/sign_language/arj/test.jpg')
frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.equalizeHist(frame_gray)

# Detect hands using classifier.
hands = hand_cascade.detectMultiScale(frame_gray)

# Draw bounding boxes round detected hands.
for (x,y,w,h) in hands:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = frame_gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
# Show image. Press any key to escape         
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



 