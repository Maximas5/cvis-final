from abc import ABC
from abc import abstractmethod

class Expert(ABC):
    def __init__(self):
        self.fWeight = None
        self.fNotWeight = None

    # @abstractmethod
    # def train(self, train):
    #     pass

    @abstractmethod
    def predict(self):
        # Returns classification of blob
        pass