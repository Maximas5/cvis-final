

class MES:
    def __init__(self, colorEx, movementEx):
        self.CE = colorEx
        self.ME = movementEx

    def train(self, train, test):
        self.CE.train(train, test)
        self.ME.train(train, test)

    def predict(self, fblob, bblob):
        pass