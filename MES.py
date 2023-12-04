

class MES:
    def __init__(self, colorEx, movementEx):
        self.CE = colorEx
        self.ME = movementEx

    def train(self, train, test):
        self.CEvalMatrix = self.CE.train(train, test)
        self.MEvalMatrix = self.ME.train(train, test)

    def predict(self, fblob, bblob):
        pass
    