#%%
#Implementing detector
import numpy as np
import cv2
from detector import Detector
from utils.draw import Draw
detector = Detector()
from classifier import classify_gesture, classify_gesture_alt

cap = cv2.VideoCapture(0)

# Set gesture dict
numbers = [x for x in range(0,24)]
letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
gesture_dict = dict(zip(numbers,letters))

# Rectangle parameters
start_point = (150,100)
end_point = (500,400)
rect_col = (255,0,0)
rect_thic = 2

while(True):
    ret, frame = cap.read()
    # Put rectangle on frame (roi guide)
    frame = cv2.rectangle(frame, start_point, end_point, rect_col, rect_thic)
    
    # Get ROI
    roi = frame[100:400, 150:500]
    # Canny edge 
    edges = cv2.Canny(roi,100,200)
    # Locate hands
    rel_boxes, scores = detector.detect_objects(cv2.cvtColor(roi, 4))
    # Working. Needs distance though.
    draw = Draw(roi, rel_boxes, scores)
    draw.bounding_box()
    hand = draw.crop(gray=True, dim=(100, 100))
    
    
    
    
    # Display frames
    cv2.imshow('frame',frame)
    cv2.imshow('roi',roi)
    cv2.imshow('edges',edges)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
#%%
# Implementing haar cascade
# Can just stick loads of them on top of each other to detect hands 
# still not as good as the neural network. 

import cv2 
import numpy as np

handb = ("./haar_classifiers/handb.xml")
aGest = ("./haar_classifiers/aGest.xml")
rpalm = ("./haar_classifiers/rpalm.xml")



hand_cascade = cv2.CascadeClassifier(handb)
hand_cascade2 = cv2.CascadeClassifier(aGest)
hand_cascade3 = cv2.CascadeClassifier(rpalm)

cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect 1
#    hands = hand_cascade.detectMultiScale(gray,1.3,5)
#    for (x1,y1,w1,h1,) in hands:
#        cv2.rectangle(img,(x1,y1,),(x1+w1,y1+h1),(250,0,0),2)
#        roi_gray = gray[y1:y1+h1,x1:x1+w1]
#        roi_color = img[y1:y1+h1, x1:x1+w1]
    # Detect 2
    hands2 = hand_cascade2.detectMultiScale(gray,1.3,5)
    for (x2,y2,w2,h2,) in hands2:
        cv2.rectangle(img,(x2,y2,),(x2+w2,y2+h2),(0,250,0),2)
        roi_gray = gray[y2:y2+h2,x2:x2+w2]
        roi_color = img[y2:y2+h2, x2:x2+w2]
    # Detect 3
    hands3 = hand_cascade3.detectMultiScale(gray,1.3,5)
    for (x3,y3,w3,h3,) in hands3:
        cv2.rectangle(img,(x3,y3,),(x3+w3,y3+h3),(0,0,250),2)
        roi_gray = gray[y3:y3+h3,x3:x3+w3]
        roi_color = img[y3:y3+h3, x3:x3+w3]
        
    
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()    