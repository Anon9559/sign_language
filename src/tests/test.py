
import unittest
import cv2 as cv

from detector import Detector
from utils.draw import Draw

class TestHandDetection(unittest.TestCase):

    def setUp(self):
        self.img = cv.imread("./tests/images/hand.png")
        self.detector = Detector()
    
    def test_hand_detection(self):
        """ Tests if a hand is present in an image, the current image being
        used requires a threshold of 0.05 to find the hand in the image
        """
        _, scores = self.detector.detect_objects(self.img)
        self.assertTrue(scores[0] > 0.05)

    def test_cropping(self):
        """ Tests to see if the region the hand is in is cropped out
        this runs the method `crop`
        """
        boxes, scores = self.detector.detect_objects(self.img)
        draw = Draw(self.img, boxes, scores, threshold=0.05)
        regions = draw.crop()
        self.assertTrue(regions)

    def test_bounding_box(self):
        """ Creates an new window and displays the image with the given bounding box
        The test has passed if there is a bounding box and the score of the detection
        """
        boxes, scores = self.detector.detect_objects(self.img)
        draw = Draw(self.img, boxes, scores, threshold=0.05)
        draw.bounding_box()
        cv.imshow("bb", self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()

