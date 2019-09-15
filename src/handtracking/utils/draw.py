
import cv2 as cv


def bounding_box(image, scores, boxes, hands=2, score=0.2, label=False):
    """ Draws a bounding box around detected hands.

    Args:
      image: image to draw bounding box on
      scores: detected hands scores (confidence)
      boxes: detected hands boundign boxes
      hands: (int) number of hands to draw
      score: (int) minimum score to draw bounding box for
    """
    height, width = image.shape[:2]
    colour = (77, 255, 9)

    for hand in range(hands):
        if (scores[hand] > score):
            left, right, top, bottom = (boxes[hand][1] * width,  boxes[hand][3] * width,
                                        boxes[hand][0] * height, boxes[hand][2] * height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv.rectangle(image, p1, p2, colour, 3, 1)
        if label:
            pass


def fps(image, n):
    """ Draws the frames per second on a given image

    Args:
      image: image to draw fps on
      fps: (int) frames per second to draw on image
    """
    cv.putText(image, str(n), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

