
import unittest

import cv2 as cv
from utils.detect import detect_objects
from utils.detect import load_inference_graph
from utils.draw import bounding_box

class TestHandDetection(unittest.TestCase):

    def setUp(self):
        self.img = cv.imread("./tests/images/hand.png")
    
    def test_hand_detection(self):
        dg, sess = load_inference_graph()
        boxes, scores = detect_objects(self.img, dg, sess)
        self.assertTrue(scores[0] > 0.05)

