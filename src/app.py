import cv2 as cv
import numpy as np
import os
import time

from keras.models import load_model
from keras.preprocessing import image
from defaults import font

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


    test_image = frame.copy()
    test_image = cv.resize(test_image, (64, 64))
    test_image = np.expand_dims(test_image, axis=0)
    
    image_class = model.predict(test_image)
    
    if image_class[0][0] == 1:
        current_gest = 'Fist'
    elif image_class[0][1] == 1:
        current_gest = 'Index'
    elif image_class[0][2] == 1:
        current_gest = 'Loser'
    elif image_class[0][3] == 1:
        current_gest = 'Okay'
    elif image_class[0][4] == 1:
        current_gest = 'Open_5'
    elif image_class[0][5] == 1:
        current_gest = 'Peace'

    k = cv.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        break

    fps_end = time.time()
    fps_ind = 1 / (fps_end  - fps_start)
            

cam.release()
cv.destroyAllWindows()
