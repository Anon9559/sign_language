#%%
#Implementing detector
import numpy as np
import cv2
from detector import Detector
from utils.draw import Draw
detector = Detector()

cap = cv2.VideoCapture(0)

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
    
    # This bit doesnt work properly. Cant seem to detect hands
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