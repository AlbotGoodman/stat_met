import numpy as np

class LinearRegression:

    def __init__(self):
        self._X = None
        self._y = None
    
    def fit(self, X, y):
        self._X = np.column_stack((np.ones(len(X)), X))
        