
import cv2 as cv
import numpy as np

from settings import FONT
from statistics import mean
from time import time

from utils.draw import bounding_box
from utils.draw import crop

from detector import Detector, MultiThreadedDetector

NUM_FRAMES = 20
fps_mean = 0
fps_times = []

cam = cv.VideoCapture(0)
cv.namedWindow("main")

detector = Detector()

while True:
    timer_start = time()

    if len(fps_times) == NUM_FRAMES:
        fps_mean = int(mean(fps_times))
        fps_times = []

    ret, frame = cam.read()
    if not ret:
        print("no camera found")
        break
    
    boxes, scores = detector.detect_objects(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
    # detect_objects(cv.cvtColor(frame, cv.COLOR_BGR2RGB), dg, sess)
    regions = crop(frame, boxes, scores, score=0.9)
    bounding_box(frame, boxes, scores)

    # gesture = classify_gesture(frame)

    # cv.putText(frame, gesture, (50, 32), FONT, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, str(fps_mean), (10, 30), FONT, 0.7, (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow("main", frame)

    timer_end = time()
    fps_times += [ round( 1 / (timer_end - timer_start), 0) ]

    k = cv.waitKey(1)

    # Exit on ESC
    if k % 256 == 27:
        break

cam.release()
cv.destroyAllWindows()

