#
# Project Defaults
#
# =============================================================================

from cv2 import FONT_HERSHEY_SIMPLEX 


# ========
# General
# ========

FONT = FONT_HERSHEY_SIMPLEX


# ===========
# Classifier
# ===========

GESTURES = [
   'Fist',
   'Index',
   'Loser',
   'Okay',
   'Open_5',
   'Peace'
]


# =========
# Detector
# =========

# hand detector inference graph
PATH_TO_CKPT = './models/frozen_inference_graph.pb'

# hand detector labels
PATH_TO_LABELS = './models/hand_label_map.pbtxt'

# hand detector number of classes
NUM_CLASSES = 1

# multi threaded hand detector number of worker in worker pool
WORKERS = 4

# multi threaded hand detector que size
QUE_SIZE = 5

