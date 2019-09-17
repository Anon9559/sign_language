
import cv2 as cv
import numpy as np

from detector import Detector
from settings import FONT
from utils.camera import VideoCaptureThreading
from utils.draw import bounding_box
from utils.draw import crop
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
        boxes, scores = detector.detect_objects(cv.cvtColor(frame, 4))

    crop(frame, boxes, scores)
    bounding_box(frame, boxes, scores, threshold=0.2, label=True)

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

