import numpy as np
import scipy.stats as stats


class LinearRegression:

    def __init__(self):
        self.b = None
        self.y_hat = None
    
    def fit(self, X, y):
        """
        Training the model.
        
        Args:
            X: Design matrix
            y: Response variable
        """
        X = np.column_stack((np.ones(len(X)), X))
        self.b = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        """
        Use the model on new data to make predictions. 
        
        Args:
            X: Design matrix
        """
        X = np.column_stack((np.ones(len(X)), X))
        self.y_hat = X @ self.b
        return self.y_hat

    def evaluate(self, X, y):
        """
        Assessing the model.
        
        Args:
            X: Design matrix
            y: Response variable
        """
        X = np.column_stack((np.ones(len(X)), X))
        d = len(self.b)-1
        n = y.shape[0]
        SSE = np.sum(np.square(y - self.y_hat))
        var = SSE / (n - d - 1)
        std_dev = np.sqrt(var)
        SST = np.sum(np.square(y - np.mean(y)))
        SSR = SST - SSE
        R_squared = SSR / SST
        F_stat = (SSR / d) / var
        F_pvalue = stats.f.sf(F_stat, d, n-d-1)
        cov_matrix = np.linalg.pinv(X.T @ X) * var
        MSE = (1 / n) * SSE
        RMSE = np.sqrt(MSE)
        t_stat = [self.b[i] / (std_dev * np.sqrt(cov_matrix[i, i])) for i in range(d)] # holds t-statistics for each coefficient
        p_values = [2 * min(stats.t.cdf(i, n-d-1), stats.t.sf(i, n-d-1)) for i in t_stat] # holds p-values for each coefficient
        return {
            "Variance": var,
            "Standard deviation": std_dev,
            "SST": SST,
            "R_squared": R_squared,
            "MSE": MSE,
            "RMSE": RMSE,
            "F_stat": F_stat,
            "F_pvalue": F_pvalue,
            "p_values": p_values
        }