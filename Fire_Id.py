import cv2
import numpy as np
from Color_Expert import Color_Expert
from Shape_Expert import Shape_Expert
from Movement_Expert import Movement_Expert
from MES import MES

class Fire_Id:
    def __init__(self):
        ce = Color_Expert()
        me = Movement_Expert()
        self.mes = MES(ce, me)

        self.backSub = None

    def get_blob(self, img):
        blobs = self.backSub.apply(img)
        blobs = cv2.connectedComponents(blobs)
        return blobs

    def get_blobs(self, imgs):
        blobs = []
        for img in imgs:
            blobs.append(self.get_blob(img))
        return blobs
    
    def predict(self, img, prevImg):
        currBlob = self.get_blob(img)
        prevBlob = self.get_blob(prevImg)
        return self.mes.predict(prevBlob, currBlob)
    
    def train(self, data):
        self.backSub = cv2.createBackgroundSubtractorKNN()

        # data [img, label]

        # For each model...
            # Run train