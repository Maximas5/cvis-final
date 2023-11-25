from abc import ABC
from abc import abstractmethod

class Expert(ABC):
    def __init__(self):
        self.weight = 0

    def train(self, train, test):
        pass

    def calc_m(self):
        pass

    @abstractmethod
    def predict(self, fblob, bblob):
        pass