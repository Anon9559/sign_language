import cv2 as cv
import numpy as np
import os
import time

from keras.models import load_model
from keras.preprocessing import image
from defaults import font
from defaults import gestures

NUM_FRAMES = 40

# import model
model = load_model("../models/gestures.h5")

# Use Webcam
cam = cv.VideoCapture(0)
cv.namedWindow("test")

fps = 0
fps_ind = 0
fps_avg = []
current_gest = "NA"

while True:

    fps_avg += [ fps_ind ]

    if len(fps_avg) == NUM_FRAMES:
        fps = round(sum(fps_avg) / NUM_FRAMES, 0)
        fpg_avg = []

    fps_start = time.time()

    ret, frame = cam.read()

    if not ret:
        print("no camera found")
        break

    # write text on image
    cv.putText(frame, current_gest, (70, 32), font, 1, (255, 255, 255), 
            2, cv.LINE_AA)
    cv.putText(frame, str(fps), (10, 30), font, 0.7, (0, 255, 255), 
            2, cv.LINE_AA)

    cv.imshow("test", frame)

    pred_image = frame.copy()
    pred_image = cv.resize(pred_image, (64, 64))
    pred_image = np.expand_dims(pred_image, axis=0)
    
    image_class = model.predict(pred_image)
    try:
        # get index of classified value and lookup in gestures list
        current_gest = gestures[image_class[0].tolist().index(1.0)]
    except ValueError:
        current_gest = "NA"

    k = cv.waitKey(1)

    # Exit on ESC
    if k % 256 == 27:
        break

    fps_end = time.time()
    fps_ind = 1 / (fps_end  - fps_start)
            

cam.release()
cv.destroyAllWindows()
