
import cv2 as cv
import numpy as np

from settings import GESTURES
from keras.models import load_model
from keras.preprocessing import image


model = load_model("./models/89.1.h5")

def classify_gesture(image):
    pred_image = image.copy()
    pred_image = np.expand_dims(pred_image, axis=0)
    pred_image = pred_image.reshape((-1, 28, 28, 1))
    image_class = model.predict(pred_image)
    
    try:
        gesture = GESTURES[image_class[0].tolist().index(1.0)]
    except ValueError:
        gesture = "NA"
    
    return gesture
#####
def classify_gesture_alt(image):
    pred_image = image.copy()
    pred_image = pred_image/255
    pred_image = pred_image.reshape(-1,28,28,1)
    image_class = model.predict(pred_image)
    
    try:
        gesture = GESTURES[image_class[0].tolist().index(1.0)]
    except ValueError:
        gesture = "NA"
    
    return gesture
