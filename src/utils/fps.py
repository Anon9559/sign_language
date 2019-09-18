
import time
import cv2 as cv
from settings import FONT

class Fps: 
    """ Measures the fps (frames per second)
    
    Example;

    >>> from utils.fps import Fps
    
    Setup

    >>> cap = cv.videoCapture()
    >>> fps = Fps()
    
    Main loop

    >>> while True:
    >>>     ret, frame = cap.read()
    >>>     fps.update()
    >>>     fps.display(frame)
    >>>     cv.imshow("main", frame)
    >>>     ...
    """
    def __init__(self):
        self.nbf=0
        self.fps=0
        self.start=0
        
    def update(self):
        """ Updates the fps reading """
        if self.nbf%10==0:
            if self.start != 0:
                self.stop=time.time()
                self.fps=10/(self.stop-self.start)
                self.start=self.stop
            else :
                self.start=time.time()    
        self.nbf+=1

    def display(self, frame, **kwargs):
        """ Displays the fps on a given frame 
        
        Kwargs:
          pos: (tuple) position of text
          font: (int) font enumerator see cv2 fonts
          size: (float) font size
          colour: (tuple) BGR font colour
          thickness: (float) text thickness
        """
        pos = kwargs.get("pos", (10,30))
        font = kwargs.get("font", FONT)
        size = kwargs.get("size", 0.7)
        colour = kwargs.get("colour", (0, 255, 255))
        thickness = kwargs.get("thickness", 2)

        cv.putText(frame, f"{self.fps:.2f}", pos, font, size, colour, thickness)
