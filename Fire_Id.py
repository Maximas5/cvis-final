import cv2
import numpy as np
from Color_Expert import Color_Expert
from Shape_Expert import Shape_Expert
from Movement_Expert import Movement_Expert
from MES import MES

class Fire_Id:
    def __init__(self):
        self.ce = Color_Expert()
        self.se = Shape_Expert()
        self.mes = None

        self.backSub = None

    def get_blobs(self, img):
        blobs = self.backSub.apply(img)
        ret, blobs = cv2.threshold(blobs, 1, 255, 0)
        blobs = cv2.connectedComponents(blobs)
        return blobs
    
    def predict(self, img, prevImg, label=False):
        '''Given an image and the previous image, return True for fire and False for no fire
        
            If label=True, return the image labeled with blobs labeled as fire
            -> labeled_img, classification

            else
            -> classification
            
        '''
        num, blobs = self.get_blobs(img)

        if num < 2:
            if label:
                return img, False
            else:
                return False

        prevBlobs = self.get_blobs(prevImg)

        # Get color image params
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yMean = np.mean(yuv[:, :, 0])
        uMean = np.mean(yuv[:, :, 1])
        vMean = np.mean(yuv[:, :, 2])

        # Get shape image params


        # For each blob...
        for blobNum in range(1, num):
            blobMask = blobs[blobs == blobNum]
            blobMask[blobMask > 0] = 1
            # Predict color
            yuvBlob = cv2.bitwise_and(yuv, yuv, mask=blobMask)
            yuvFormatted = []

            for i in range(3):
                yuvFormatted = yuvBlob[:, :, i].flatten()

            Cce = self.ce.predict(yuvFormatted, yMean, uMean, vMean)

            # Predict shape
            blob = cv2.bitwise_and(img, img, mask=blobMask)
            blobMatch = self.match(blobMask, prevBlobs)
            if blobMatch != None:
                prevMask = prevBlobs[prevBlobs == blobMatch]
                prevMask[blobMask > 0] = 1
                blobMatch = cv2.bitwise_and(prevImg, prevImg, mask=prevMask)

            Cse = self.se.predict(blob, blobMatch)

            # Predict MES
            Cmes = self.mes.predict(Cce, Cse)

            # If label, label blob with classification of experts
            if label:
                # TODO: probably won't be done by due date
                pass

        # if label, return labeled image and classification
        if label:   
            # TODO: probably won't be done by due date
            return img, Cmes
        return Cmes
    
    # TODO: Likely won't be done by due date. Have to at least train the backsub.
    def train(self, data):
        self.backSub = cv2.createBackgroundSubtractorKNN()
        self.mes = MES([1,1], [1,1])
        
        for img in data:
            self.backSub.apply(img)

    # TODO: This needs to be done for the rest of the program to function
    def match(self, blobmask, prevBlobs):
        '''Determines if the current blob overlaps with another blob from the previous frame
        
            -> label: int
        '''
        # Apply mask to prevBlobs
        prevSelection = cv2.bitwise_and(prevBlobs, prevBlobs, mask=blobmask)
        # Find unique
        unique = np.unique(prevSelection, return_counts=True)
        # Get highest quantity
        argmax = np.argmax(unique[1])
        blob = unique[0][argmax]
        if blob == 0:
            return None

        return blob