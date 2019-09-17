
import time
import cv2 as cv
from settings import FONT

class Fps: 
    # To measure the number of frame per second

    def __init__(self):
        self.nbf=0
        self.fps=0
        self.start=0
        
    def update(self):
        if self.nbf%10==0:
            if self.start != 0:
                self.stop=time.time()
                self.fps=10/(self.stop-self.start)
                self.start=self.stop
            else :
                self.start=time.time()    
        self.nbf+=1
    
    def get(self):
        return self.fps

    def display(self, frame, orig=(10,30), font=FONT, size=0.7, color=(0,255, 255), thickness = 2):
        cv.putText(frame, f"{self.get():.2f}", orig, font, size, color, thickness)
