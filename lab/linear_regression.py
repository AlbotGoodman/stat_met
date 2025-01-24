import numpy as np
import scipy.stats as stats

class LinearRegression:

    def __init__(self):
        self.b = None
        self.k = None
        self.n = None
        self.SSE = None
        self.var = None
        self.S = None
        self.Syy = None
        self.SSR = None
        self.Rsq = None
        self.sig_stat = None
        self.p_sig = None
        self.c = None
    
    def fit(self, X, y):
        """
        Info
        
        Args:
            X: 
            y: 
        """
        X = np.column_stack((np.ones(len(X)), X))
        self.b = np.linalg.inv(X.T @ X) @ X.T @ y
        self.k = len(self.b)-1
        self.n = y.shape[0]
    
    def predict(self, X):
        """
        Info
        
        Args:
            X:
        """
        pass

    def evaluate(self, X, y):
        """
        Info
        
        Args:
            X:
            y:
        """
        pass