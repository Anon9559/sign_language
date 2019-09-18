
import unittest
import cv2 as cv

from detector import Detector

class TestHandDetection(unittest.TestCase):

    def setUp(self):
        self.img = cv.imread("./tests/images/hand.png")
    
    def test_hand_detection(self):
        detector = Detector()
        _, scores = detector.detect_objects(self.img)
        self.assertTrue(scores[0] > 0.05)




