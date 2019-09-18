
import cv2 as cv
import numpy as np
import tensorflow as tf

from multiprocessing import Queue, Pool
from settings import PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, WORKERS, QUE_SIZE
from utils import label_map


# load label map
labels = label_map.load_labelmap(PATH_TO_LABELS)
categories = label_map.convert_label_map_to_categories(
        labels, 
        max_num_classes=NUM_CLASSES, 
        use_display_name=True)
category_index = label_map.create_category_index(categories)

class Detector:
    """ Locates hands within an image 
    
    Example:

    >>> import cv2 as cv
    >>> from detector import Detector
   
    Camera input

    >>> cap = cv.VideoCapture(0)
    >>> cv.namedWindow('Hand Detection')
    >>> detector = Detector()

    main loop

    >>> while True:
    >>>     ret, frame = cap.read()
    >>>     boxes, scores = detector.detect_objects(cv.cvtColor(frame, 4))
    >>>     # ...
    >>>     cv.imshow('Hand Detection', frame)
   
    break condition

    >>>     if cv.waitKey(1) & 0xFF == ord('q'):
    >>>         break
   
    Cleanup

    >>> cap.release()
    >>> cv.destroyAllWindows()
    """

    def __init__(self):
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()

            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.session = tf.compat.v1.Session(graph=self.graph)


        self._image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self._boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self._scores = self.graph.get_tensor_by_name('detection_scores:0')
        self._classes = self.graph.get_tensor_by_name('detection_classes:0')
        self._num = self.graph.get_tensor_by_name('num_detections:0')

    def detect_objects(self, image):
        """ Detects object using a given detection graph

        Args:
          image: (ndarray) input image to use.

        Returns:
            a tuple with the bounding boxes of detected hands and the 
            score (confidence) the region contains a hand
        """
        pred_image = np.expand_dims(image, axis=0)
        boxes, scores, classes, num = self.session.run([
            self._boxes, 
            self._scores,
            self._classes, 
            self._num],
            feed_dict={ self._image_tensor: pred_image }
        )
        return np.squeeze(boxes), np.squeeze(scores)

    def close_session(self):
        """ used for multi threaded detector to close the session """
        self.session.close()


def _worker(input_q, output_q):
    """ worker method, to detect hands, takes input images (ndarray) and
    passes processed output to a Queue 
    
    Args:
      input_q: (Queue) input que for images
      output_q: (Queue) output que for processed results
    """
    detector = Detector()

    while True:
        image = input_q.get()
        if (image is not None):
            boxes, scores = detector.detect_objects(image)
            output_q.put((boxes, scores))

    detector.close_session()


class MultiThreadedDetector:
    """ Mulithreaded Hand Detector

    More efficient implementation of the Detector class that calls upon
    multiple threads in order to detect hands. There are two correspondsing
    settings variables that can be altered to affect how the class works

    Settings:
      WORKER: the number of workers
      QUE_SIZE: the maximum que size for a given input / output que

    Example:

    >>> import cv2 as cv
    >>> from detector import MultiThreadedDetector

    created a threaded camera input

    >>> cap = cv.VideoCapture(0)
    >>> cv.namedWindow('Multi-Threaded Detection')

    new instance of class

    >>> mt_detector = MultiThreadedDetector()

    main loop

    >>> while True:
    >>>     frame = cap.read()
    >>>     boxes, scores = mt_detector.detect_objects(cv.cvtColor(frame, 4))
    >>>     # ...
    >>>     cv.imshow('Multi-Threaded Detection', frame)
   
    break condition

    >>>     if cv.waitKey(1) & 0xFF == ord('q'):
    >>>         break
   
    Cleanup

    >>> mt_detector.terminate()
    >>> cap.release()
    >>> cv.destroyAllWindows()

    """

    def __init__(self):
        self.input_q = Queue(maxsize=QUE_SIZE)
        self.output_q = Queue(maxsize=QUE_SIZE)
        self.pool = Pool(WORKERS, _worker, (self.input_q, self.output_q))
    
    def detect_objects(self, image):
        """ Passes an image to the worker function and returns detected objects

        Args:
          image: (ndarray) image to pass to the worker function

        Returns:
          Tuple of lists the first being of the found boxes which is a list of the
          form (l, r, t, b) with the relative positions in the image of the
          detected hand, and the second is the score, or confidence a hand is
          present
        """
        self.input_q.put(image)
        return self.output_q.get()

    def terminate(self):
        self.pool.terminate()

