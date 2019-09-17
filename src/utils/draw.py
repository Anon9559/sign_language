
import cv2 as cv
import numpy as np

from settings import FONT


def _points(box, width, height):
    """ Extract points from relative point locations of a hand

    Args:
      box: (list) points of relative object location
      width: (int) image width
      height: (int) image height

    Returns:
      A tuple with the object points
    """
    x1, x2, y1, y2 = (int(box[1] * width),
                      int(box[3] * width),
                      int(box[0] * height), 
                      int(box[2] * height))
    return x1, x2, y1, y2



def bounding_box(image, boxes, scores, threshold=0.5, **kwargs):
    """ Draws a bounding box around detected hands.

    Args:
      image: (ndarray) image to draw bounding box on
      boxes: (list) detected hands boundign boxes
      scores: (list) detected hands scores (confidence)
      threshold: (int) minimum score to draw bounding box for

    Kwargs:
      colour: (tuple) colour of bounding box
      label: (bool) display label of score
      pm: (int) plus minus border, increase box size
    """
    colour = kwargs.get("colour", (255, 0, 100))
    label  = kwargs.get("label", False)
    pm = kwargs.get("pm", 30)

    height, width = image.shape[:2]

    for box, score in zip(boxes, scores):
        if (score > threshold):
            x1, x2, y1, y2  = _points(box, width, height)
            p1 = (x1 - pm, y1 - pm)
            p2 = (x2 + pm, y2 + pm)
            cv.rectangle(image, p1, p2, colour, 3, 1)
            if label:
                pos = (int((x1 + x2)/2 - 10), y2 + pm + 30)
                cv.putText(image, f"{score:.2f}", pos, FONT, 0.5, (255, 255, 255), 2)


def crop(image, boxes, scores, threshold=0.2, **kwargs):
    """ Crops 

    Args:
      image: (ndarray) image to crop
      boxes: (list) detected object relative location
      scores: (list) detected object scores (confidence)
      threshold: (int) minimum score to select object

    Returns:
        (list) (ndarray) of cropped regions
    """
    pm = kwargs.get("pm", 5)
    dim = kwargs.get("dim", (96, 96))

    height, width = image.shape[:2]
    regions = []
    for box, score in zip(boxes, scores):
        if (score > threshold):
            x1, x2, y1, y2  = _points(box, width, height)
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > width:  x2 = width
            if y2 > height: y2 = height

            region = image[y1:y2, x1:x2]
            regions += [region]
            # region = cv.medianBlur(region, 5)
            region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
            # region = cv.resize(region, (28, 28)) # resize to match CNN input
            region = cv.resize(region, dim) # resize to match CNN input
    
    return regions

