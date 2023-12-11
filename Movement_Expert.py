'''
This expert will likely not be implemented by the time of due date.
'''

from Expert import Expert
import cv2

class Movement_Expert(Expert):
    def __init__(self):
        super().__init__()
        self.sift = cv2.SIFT.create()

    def predict(self, blob, prevImg):
        # Detect corners (using some other paper's work found here: https://link.springer.com/chapter/10.1007/11744023_34)

        # Measure Gradients (SIFT)

        # Match corners of blob to previous image

        # Get angles between matched corners and describe

        # Compute histogram

        # Measure "Homogeneity" (24)

        # Classify

        pass
