import cv2 as cv
import numpy as np
import os

from keras.models import load_model
from keras.preprocessing import image

# import model
constant_path = "./arj_legacy/"
model = load_model('arj_model_3.h5')

# Use Webcam
cam = cv.VideoCapture(0)
cv.namedWindow("test")

font = cv.FONT_HERSHEY_SIMPLEX

img_counter = 0
current_gest = ""

while True:
    ret, frame = cam.read()
    cv.putText(frame, current_gest, (10, 50), font, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("test", frame)

    if not ret:
        break
    k = cv.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
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
        else:
            current_gest = 'not found'
            

cam.release()

cv.destroyAllWindows()

