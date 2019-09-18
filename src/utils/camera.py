
import threading
import cv2 as cv


class VideoCaptureThreading:
    """ Threaded Video Capture, uses an new thread to fetch frames. Has the effect
    of increasing the frame rate substantually.

    Example:

    >>> from utils.camera import VideoCaptureThreading

    >>> cap = VideoCaptureThreading()
    >>> cap.start()

    Main loop

    >>> while True:
    >>>     ret, frame = cap.read()
    >>>     cv.imshow("main", frame)
    >>> 
    >>>     k = cv.waitKey(1)
    >>>     if k % 256 == 27:
    >>>         break

    Tidy up

    >>>     cap.stop()
    >>>     cv.destroyAllWindows()
    """
    def __init__(self, src=0, width=640, height=480):
        """ Initalisation

        Args:
          src: (int) input source
          width: (int) width of frame
          height: (int) height of frame
        """
        self.src = src
        self.cap = cv.VideoCapture(self.src)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        """ Interact with OpenCV VideoCapture Class 
        
        Args:
          var1: (int) camera value to set
          var2: value to set
        """
        self.cap.set(var1, var2)

    def start(self):
        """ Starts the thread. The thread is running the update method
        
        Returns:
          Instance object
        """
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        """ Updates the frame """
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        """ Reads Camera input
        
        Returns:
          (tuple) containing the state of the camera input and the frame
        """
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        """ Stops reading camera input """
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
