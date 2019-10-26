
import cv2 as cv
import numpy as np

from settings import FONT


class Draw:
    """ Takes images and detected objects and applys operations to them

    Args:
      image: (ndarray) input image
      rel_boxes: (list) location of relative objects within the image
      scores: (list) confidence that the given object is within object class

    Example:

    >>> import cv2 as cv
    >>> from utils.draw import Draw

    Setup

    >>> img = cv.imread("testimage.jpg")
    >>> rel_bounding_boxes = [[0.3, 0.5, 0.3, 0.5], [0.8, 0.85, 0.8, 0.85]]
    >>> scores = [ 0.95, 0.90 ]

    >>> draw = Draw(img, rel_bounding_boxes, scores)

    Apply draw functions

    >>> draw.bounding_box()
    >>> objs = draw.crop()

    Show images

    >>> cv.imshow("image", img)
    """

    def __init__(self, image, rel_boxes, scores, threshold=0.5, hands=2):
        self.threshold = threshold
        self.image = image
        self.height, self.width = image.shape[:2]
        self.scores = scores[:hands]
        self.rel_boxes = rel_boxes[:hands]
        self.boxes = self._boxes_points()

    def _boxes_points(self):
        """ Converts the relative bounding boxes of detected object to 
        the real points within an image
        
        Returns:
          (list) of (tuple) containing the real points of detected objects
        """
        boxes = []
        for box, score in zip(self.rel_boxes, self.scores):
            if (score > self.threshold):
                x1, x2, y1, y2 = (int(box[1] * self.width),
                                  int(box[3] * self.width),
                                  int(box[0] * self.height), 
                                  int(box[2] * self.height))
                boxes += [(x1, x2, y1, y2)]
        return boxes
                
    def bounding_box(self, **kwargs):
        """ Draws a bounding box around detected hands.

        Kwargs:
          colour: (tuple) colour of bounding box
          label: (bool) display label of score
          pm: (int) plus minus border, increase box size
        """
        colour = kwargs.get("colour", (255, 0, 100))
        label  = kwargs.get("label", True)
        pm = kwargs.get("pm", 30)

        for (x1, x2, y1, y2), score in zip(self.boxes, self.scores):
            p1 = (x1 - pm, y1 - pm)
            p2 = (x2 + pm, y2 + pm)
            cv.rectangle(self.image, p1, p2, colour, 3, 1)
            if label:
                pos = (int((x1 + x2)/2 - 10), y2 + pm + 30)
                cv.putText(self.image, f"{score:.2f}", pos, FONT, 0.5, (255, 255, 255), 2)


    def crop(self, **kwargs):
        """ Crops 

        Kwargs:
          pm: (int) plus minus border, increase box size
          dim: (tuple) dimension to resize crop to
          gray: (bool) convert image to grayscale

        Returns:
            (list) (ndarray) of cropped regions
        """
        gray = kwargs.get("gray", False)
        pm = kwargs.get("pm", 5)
        dim = kwargs.get("dim", False)

        regions = []
        for x1, x2, y1, y2 in self.boxes:
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > self.width:  x2 = self.width
            if y2 > self.height: y2 = self.height

            region = self.image[y1:y2, x1:x2]
            if gray: region = cv.cvtColor(region, 6)
            if dim: region = cv.resize(region, dim)

            regions += [region]
        
        return regions

