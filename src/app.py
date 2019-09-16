
import cv2 as cv
import numpy as np

from defaults import font
from defaults import gestures
from keras.models import load_model
from keras.preprocessing import image
from statistics import mean
from time import time

from handtracking.utils.detect import load_inference_graph
from handtracking.utils.detect import detect_objects
from handtracking.utils.draw import bounding_box
from handtracking.utils.draw import crop

NUM_FRAMES = 20
fps_mean = 0
fps_times = []

model = load_model("../models/model_14_09.h5")

def classify_gesture(frame):
    pred_image = frame.copy()
    cv.cvtColor(pred_image, cv.COLOR_BGR2RGB)
    pred_image = cv.resize(pred_image, (64, 64))
    pred_image = np.expand_dims(pred_image, axis=0)
    
    image_class = model.predict(pred_image)
    try:
        # finds index of classified gesture
        gesture = gestures[image_class[0].tolist().index(1.0)]
    except ValueError:
        gesture = "NA"
    
    return gesture


cam = cv.VideoCapture(0)
cv.namedWindow("test")

dg, sess = load_inference_graph()

while True:
    timer_start = time()

    if len(fps_times) == NUM_FRAMES:
        fps_mean = int(mean(fps_times))
        fps_times = []

    ret, frame = cam.read()
    if not ret:
        print("no camera found")
        break
    
    boxes, scores = detect_objects(cv.cvtColor(frame, cv.COLOR_BGR2RGB), dg, sess)
    bounding_box(frame, boxes, scores)
    regions = crop(frame, boxes, scores, score=0.9)

    gesture = classify_gesture(frame)

    cv.putText(frame, gesture, (50, 32), font, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, str(fps_mean), (10, 30), font, 0.7, (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow("test", frame)

    timer_end = time()
    fps_times += [ round( 1 / (timer_end - timer_start), 0) ]

    k = cv.waitKey(1)

    # Exit on ESC
    if k % 256 == 27:
        break

cam.release()
cv.destroyAllWindows()

