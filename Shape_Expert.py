from Expert import Expert
import numpy as np
import cv2

class Shape_Expert(Expert):
    def __init__(self):
        super().__init__()
        self.thresh = 40

    def predict(self, blob, prevBlob, verbose=False):
        '''Assumes blob has been matched to previous blob. Also assumes blobs are binary images.'''
        if prevBlob == None:
            return False
        # Compute perimeter-area ratios
        r = self.calc_rt(blob)
        r1 = self.calc_rt(prevBlob)

        # Compute Shape variation
        s = abs((r - r1) / r)

        # Classify
        C = s > self.thresh

        # Return the class
        return C
    
    # TODO: Needs to be completed for program to function
    def calc_rt(self, blob):
        # Code robbed from here: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
        # Get contours
        ret, thresh = cv2.threshold(blob,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]

        # Get area
        area = cv2.contourArea(cnt)

        # Get perimeter
        perimeter = cv2.arcLength(cnt,True)

        # Calculate r
        r = perimeter / area

        return r
    
    def print(self, message):
        print(f"Shape_Expert.py: {message}")