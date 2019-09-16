
import cv2 as cv
from src.handtracking.utils.detect import detect_objects
from src.handtracking.utils.detect import load_inference_graph
from src.handtracking.utils.draw import bounding_box

img = cv.imread("images/hand.jpg")

dg, sess = load_inference_graph()
boxes, scores = detect_objects(image, detection_graph, sess)
img = bounding_box(image, scores, boxes)

cv.imshow("test", img)

cv.waitKey(0)
cv.destroyAllWindows()
