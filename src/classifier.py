
import cv2 as cv
import numpy as np

from settings import GESTURES
from keras.models import load_model
from keras.preprocessing import image


model = load_model("./models/model_14_09.h5")

def classify_gesture(image):
    pred_image = image.copy()
    pred_image = cv.cvtColor(pred_image, cv.COLOR_BGR2RGB)
    pred_image = cv.resize(pred_image, (64, 64))
    pred_image = np.expand_dims(pred_image, axis=0)
    
    image_class = model.predict(pred_image)
    
    try:
        gesture = GESTURES[image_class[0].tolist().index(1.0)]
    except ValueError:
        gesture = "NA"
    
    return gesture

