import cv2 as cv
import numpy as np

from detector import Detector
from classifier import classify_gesture, classify_gesture_alt
from settings import FONT
from utils.camera import VideoCaptureThreading
from utils.draw import Draw
from utils.fps import Fps

#########
# Not sure where to put this, sorry Mike
numbers = [x for x in range(0,24)]
letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
gesture_dict = dict(zip(numbers,letters))
#########

cap = VideoCaptureThreading()
cap.start()
fps = Fps()
detector = Detector()

cv.namedWindow("main")

while True:
    ret, frame = cap.read()

    if not ret:
        print("no camera found")
        break

    # frame = cv.medianBlur(frame, 3)

    if fps.nbf % 2 == 0:
        rel_boxes, scores = detector.detect_objects(cv.cvtColor(frame, 4))


    draw = Draw(frame, rel_boxes, scores)
    draw.bounding_box()
    roi = draw.crop(gray=True, dim=(28, 28))
    
    if len(roi) > 0:
        hand = roi[0]
        gesture_key = classify_gesture_alt(hand)
        try:
            gesture_key = int(gesture_key)
            gesture_value = gesture_dict[gesture_key]
        except:
            gesture_value = 'NA'
            
        cv.putText(frame, gesture_value, (80, 32), FONT, 1, (255, 255, 255), 2, cv.LINE_AA)

    fps.update()
    fps.display(frame)

    cv.imshow("main", frame)

    k = cv.waitKey(1)
    # Exit on ESC
    if k % 256 == 27:
        break

cap.stop()
cv.destroyAllWindows()


