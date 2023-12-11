from Expert import Expert
import numpy as np
import cv2

class Shape_Expert(Expert):
    def __init__(self):
        super().__init__()
        self.thresh = 40

    def predict(self, blob, prevBlob):
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
        # Get contours

        # Get area

        # Get perimeter

        # Calculate r

        return r