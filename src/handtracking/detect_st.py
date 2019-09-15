
import cv2 as cv
import tensorflow as tf

from time import time
from utils.detect import detect_objects
from utils.detect import load_inference_graph
from utils.draw import bounding_box

VIDEO_SOURCE = 0

detection_graph, sess = load_inference_graph()

cap = cv.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    try:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

    boxes, scores = detect_objects(frame, detection_graph, sess)
    bounding_box(frame, scores, boxes) # draw boxes

    cv.imshow('Single-Threaded Detection', cv.cvtColor(frame, cv.COLOR_RGB2BGR))

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
