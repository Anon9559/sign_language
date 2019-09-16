
import cv2 as cv
import numpy as np

def bounding_box(image, boxes, scores, hands=2, score=0.2, label=False, pm=30):
    """ Draws a bounding box around detected hands.

    Args:
      image: image to draw bounding box on
      boxes: detected hands boundign boxes
      scores: detected hands scores (confidence)
      hands: (int) number of hands to draw
      score: (int) minimum score to draw bounding box for
    """
    height, width = image.shape[:2]
    colour = (77, 255, 9)

    for hand in range(hands):
        if (scores[hand] > score):
            l, r, t, b = (boxes[hand][1] *  width - pm, boxes[hand][3] * width + pm,
                          boxes[hand][0] * height - pm, boxes[hand][2] * height + pm)
            p1 = (int(l), int(t))
            p2 = (int(r), int(b))
            cv.rectangle(image, p1, p2, colour, 3, 1)
        if label:
            pass


def crop(image, boxes, scores, hands=2, score=0.2, label=False, pm=30):
    """ Crops 

    Args:
      image: image to draw bounding box on
      boxes: detected hands boundign boxes
      scores: detected hands scores (confidence)
      hands: (int) number of hands to draw
      score: (int) minimum score to draw bounding box for

    Returns:
        List of cropped regions (ndarray) 
    """
    height, width = image.shape[:2]
    regions = []
    for hand in range(hands):
        if (scores[hand] > score):
            x1, x2, y1, y2 = (int(boxes[hand][1] * width - pm),
                              int(boxes[hand][3] * width + pm),
                              int(boxes[hand][0] * height - pm), 
                              int(boxes[hand][2] * height + pm))
            if x1 < 0:
                x1 = 0
            if x2 > width:
                x2 = width
            if y1 < 0:
                y1 = 0
            if y2 > height:
                y2 = height

            region = image[y1:y2, x1:x2]
            regions += [region]
            region = cv.medianBlur(region, 5)
            region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
            # region = cv.resize(region, (28, 28)) # resize to match CNN input
            cv.imshow(str(hand), region)
    
    return regions


def fps(image, n):
    """ Draws the frames per second on a given image

    Args:
      image: image to draw fps on
      fps: (int) frames per second to draw on image
    """
    cv.putText(image, str(n), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

