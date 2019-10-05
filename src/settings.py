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

GESTURES = [str(x) for x in range(0,25)]


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
WORKERS = 2

# multi threaded hand detector que size
QUE_SIZE = 5

