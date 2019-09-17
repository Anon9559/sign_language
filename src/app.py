
import cv2 as cv
import numpy as np

from detector import Detector
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

    if fps.nbf % 2 == 0:
        rel_boxes, scores = detector.detect_objects(cv.cvtColor(frame, 4))
    
    draw = Draw(frame, rel_boxes, scores)
    draw.bounding_box()
    draw.crop()

    # gesture = classify_gesture(frame)
    # cv.putText(frame, gesture, (50, 32), FONT, 1, (255, 255, 255), 2, cv.LINE_AA)

    fps.update()
    fps.display(frame)

    cv.imshow("main", frame)

    k = cv.waitKey(1)
    # Exit on ESC
    if k % 256 == 27:
        break

cap.stop()
cv.destroyAllWindows()

