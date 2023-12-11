'''Class responsible for combining classifications of each expert (done per blob)'''
import numpy as np

class MES:
    def __init__(self, CE_Weights, SE_Weights):
        self.CE = CE_Weights
        self.SE = SE_Weights

    def predict(self, Cce, Cse):
        '''Takes the class predictions of each experts and combines them into one final class prediction'''
        # If both are false will return false, otherwise it will return true
        return np.max([Cce, Cse])
    
    def print(self, message):
        print(f"MES.py: {message}")