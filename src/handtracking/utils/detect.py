# Utilities for object detector.

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import label_map


detection_graph = tf.Graph()

# score threshold for showing bounding boxes.

PATH_TO_CKPT = './models/frozen_inference_graph.pb'
PATH_TO_LABELS = './models/hand_label_map.pbtxt'
NUM_CLASSES = 1

# load label map
labels = label_map.load_labelmap(PATH_TO_LABELS)
categories = label_map.convert_label_map_to_categories(
        labels, 
        max_num_classes=NUM_CLASSES, 
        use_display_name=True)
category_index = label_map.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    detection_graph = tf.compat.v1.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    return detection_graph, sess



def detect_objects(frame, detection_graph, sess):
    ''' Detects object using a given detection graph

    Args:
      frame: image
      detection_graph: inference graph
      sess: ?

    Returns:
        a tuple with the bounding boxes of detected hands and the score (confidence)
    the region contains a hand
    '''
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') # roi
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') # confidence
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_expanded = np.expand_dims(frame, axis=0)

    boxes, scores, classes, num = sess.run([
            detection_boxes, 
            detection_scores,
            detection_classes, 
            num_detections
            ],
            feed_dict={ image_tensor: image_expanded }
    )

    return np.squeeze(boxes), np.squeeze(scores)

