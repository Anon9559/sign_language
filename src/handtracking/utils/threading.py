
import tensorflow as tf

from utils.detect import detect_objects
from utils.detect import load_inference_graph
from utils.draw import bounding_box

def worker(input_q, output_q, cap_params, frame_processed):
    detection_graph, sess = load_inference_graph()
    sess = tf.compat.v1.Session(graph=detection_graph)

    while True:
        frame = input_q.get()
        if (frame is not None):
            boxes, scores = detect_objects(frame, detection_graph, sess)
            bounding_box(frame, scores, boxes)
            output_q.put(frame) # add annotaded frame
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()
