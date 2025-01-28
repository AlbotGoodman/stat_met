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
        self.b = np.linalg.pinv(X.T @ X) @ X.T @ y      # an Ordinary Least Squares method
    
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
        d = len(self.b)-1                           # number of parameters/dimensions/features
        n = y.shape[0]
        SSE = np.sum(np.square(y - self.y_hat))
        var = SSE / (n - d - 1)
        std_dev = np.sqrt(var)
        SST = np.sum(np.square(y - np.mean(y)))
        SSR = SST - SSE
        R_squared = SSR / SST
        r = stats.pearsonr(y, self.y_hat)[0]
        MSE = (1 / n) * SSE
        RMSE = np.sqrt(MSE)
        f_stat = (SSR / d) / var
        f_pvalue = stats.f.sf(f_stat, d, n-d-1)         # tests significance of all parameters at once
        cov_matrix = np.linalg.pinv(X.T @ X) * var
        ti_stat = [self.b[i] / (std_dev * np.sqrt(cov_matrix[i, i])) for i in range(d)] # holds t-statistics for each coefficient
        ti_pvalues = [2 * min(stats.t.cdf(t, n-d-1), stats.t.sf(t, n-d-1)) for t in ti_stat] # holds p-values for each coefficient
        return {
            "Variance": var,
            "Standard deviation": std_dev,
            "SSE": SSE,
            "SST": SST,
            "SSR": SSR,
            "R_squared": R_squared,
            "r": r,
            "MSE": MSE,
            "RMSE": RMSE,
            "F_stat": f_stat,
            "F_pvalue": f_pvalue,
            "cov_matrix": cov_matrix,
            "ti_stat": ti_stat,
            "ti_pvalues": ti_pvalues
        }