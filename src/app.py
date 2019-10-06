
import cv2 as cv
import numpy as np

from detector import Detector
from classifier import classify_gesture
from settings import FONT
from utils.camera import VideoCaptureThreading
from utils.draw import Draw
from utils.fps import Fps


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
        gesture = classify_gesture(hand)
        cv.putText(frame, gesture, (50, 32), FONT, 1, (255, 255, 255), 2, cv.LINE_AA)

     # Gets in way of prediction. Removed.
#    fps.update()
#    fps.display(frame)

    cv.imshow("main", frame)

    k = cv.waitKey(1)
    # Exit on ESC
    if k % 256 == 27:
        break

cap.stop()
cv.destroyAllWindows()

