
import cv2 as cv

from multiprocessing import Queue, Pool
from utils.detect import load_inference_graph
from utils.detect import detect_objects
from utils.draw import bounding_box
from utils.camera import WebcamVideoStream
from utils.threading import worker

WORKERS = 4
QUE_SIZE = 5
score_thresh = 0.2
frame_processed = 0
cap_params = {}

cap = WebcamVideoStream(src=0, width=200, height=300).start()
cap_params['im_width'], cap_params['im_height'] = cap.size()
cap_params['score_thresh'] = score_thresh
cap_params['num_hands_detect'] = 2

# paralleize detection.
input_q = Queue(maxsize=QUE_SIZE)
output_q = Queue(maxsize=QUE_SIZE)

pool = Pool(WORKERS, worker, (input_q, output_q, cap_params, frame_processed))

cv.namedWindow('Multi-Threaded Detection')

while True:
    frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.flip(frame, 1)

    input_q.put(frame)

    output_frame = output_q.get()
    output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)
    cv.imshow('Multi-Threaded Detection', output_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

pool.terminate()
cap.stop()
cv.destroyAllWindows()

