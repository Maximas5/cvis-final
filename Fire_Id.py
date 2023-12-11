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
    
    def predict(self, img, prevImg, label=False, verbose=False):
        '''Given an image and the previous image, return True for fire and False for no fire
        
            If label=True, return the image labeled with blobs labeled as fire
            -> labeled_img, classification

            else
            -> classification
            
        '''
        if verbose:
            self.print("Getting blobs...")
        num, blobs = self.get_blobs(img)

        if num < 2:
            if verbose:
                self.print("No Objects found")
            if label:
                return img, False, False, False
            else:
                return False, False, False

        prevNum, prevBlobs = self.get_blobs(prevImg)

        # Get color image params
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yMean = np.mean(yuv[:, :, 0])
        uMean = np.mean(yuv[:, :, 1])
        vMean = np.mean(yuv[:, :, 2])

        totalCce = False
        totalCse = False
        totalCmes = False

        # For each blob...
        for blobNum in range(1, num):
            blobMask = self.get_mask(blobs, blobNum)

            # Predict color
            yuvBlob = self.apply_mask(yuv, blobMask)

            yuvFormatted = []
            for i in range(3):
                yuvFormatted.append(yuvBlob[:, :, i].flatten())

            if verbose:
                self.print("Predicting color...")
            Cce = self.ce.predict(yuvFormatted, yMean, uMean, vMean, verbose=verbose)
            if verbose:
                self.print(f'Class found by color: {Cce}')

            # Predict shape
            if verbose:
                self.print("Predicting shape...")
            blob = self.apply_mask(img, blobMask)
            blobMatch = self.match(blobMask, prevBlobs)
            if blobMatch != None:
                prevMask = self.get_mask(prevBlobs, blobMatch)
                blobMatch = self.apply_mask(prevImg, prevMask)

            Cse = self.se.predict(blob, blobMatch, verbose=verbose)
            if verbose:
                self.print(f'Class found by shape: {Cse}')

            # Predict MES
            Cmes = self.mes.predict(Cce, Cse)

            if Cce:
                totalCce = True
            if Cse:
                totalCse = True
            if Cmes:
                totalCmes = True

            # If label, label blob with classification of experts
            if label:
                # TODO: probably won't be done by due date
                pass

        # if label, return labeled image and classification
        if label:   
            # TODO: probably won't be done by due date
            return img, totalCce, totalCse, totalCmes
        return totalCce, totalCse, totalCmes
    
    # TODO: Likely won't be done by due date. Have to at least train the backsub.
    def train(self, data):
        self.backSub = cv2.createBackgroundSubtractorKNN()
        # Usually the model would be trained and provided weights based on "correctness", but I'm bad
        # at time management
        self.mes = MES([1,1], [1,1])
        
        for img in data:
            self.backSub.apply(img)

    def match(self, blobmask, prevBlobs):
        '''Determines if the current blob overlaps with another blob from the previous frame
        
            -> label: int
        '''
        # Apply mask to prevBlobs
        prevSelection = self.apply_mask(prevBlobs, blobmask)
        # Find unique
        unique = np.unique(prevSelection, return_counts=True)
        # Get highest quantity
        argmax = np.argmax(unique[1])
        blob = unique[0][argmax]
        if blob == 0:
            return None

        return blob
    
    def get_mask(self, labels, selector):
        mask = labels.copy()
        mask[mask != selector] = 0
        mask[mask > 0] = 1
        return mask.astype(np.uint8)
    
    def apply_mask(self, img, mask):
        return cv2.bitwise_and(img, img, mask=mask)
    
    def print(self, message):
        print(f"Fire_Id.py: {message}")