

class MES:
    def __init__(self, colorEx, movementEx):
        self.CE = colorEx
        self.ME = movementEx

        self.CEF = None
        self.CEFNot = None

        self.MEF = None
        self.MEFNot = None

    def train(self, data):
        self.CEF, self.CEFNot = self.CE.train(data)
        self.MEF, self.MEFNot = self.ME.train(data)

    # Takes the blobs in and returns a class prediction (0 or 1)
    def predict(self, img, prev_img, blob, prev_blob):
        pass