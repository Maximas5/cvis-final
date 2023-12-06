

class MES:
    def __init__(self, colorEx, movementEx):
        self.CE = colorEx
        self.ME = movementEx

    def train(self, data):
        self.CEvalMatrix = self.CE.train(data)
        self.MEvalMatrix = self.ME.train(data)

    # Takes the blobs in and returns a class prediction (0 or 1)
    def predict(self, blobT, blobT1):
        pass