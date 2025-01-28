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
        self.b = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        """
        Use the model on new data to make predictions. 
        
        Args:
            X: Design matrix
        """
        y_hat = X @ self.b
        return y_hat

    def evaluate(self, X, y):
        """
        Assessing the model.
        
        Args:
            X: Design matrix
            y: Response variable
        """
        
        k = len(self.b)-1
        n = y.shape[0]
        SSE = np.sum(np.square(y - self.y_hat))
        variance = SSE / (n - k - 1)
        std_err = np.sqrt(self.variance)
        Syy = (n * np.sum(np.square(y)) - np.square(np.sum(y)))/n
        SSR = Syy - SSE
        R_squared = SSR / Syy
        F_stat = (SSR / k) / std_err
        F_pvalue = stats.f.sf(F_stat, k, n-k-1)
        cov_matrix = np.linalg.pinv(X.T @ X) * variance
        RMSE = np.sqrt((1/(n-2))*SSE)
        t_stat = [self.b[i] / (std_err * np.sqrt(cov_matrix[i, i])) for i in range(k)] # holds t-statistics for each coefficient
        p_values = [2 * min(stats.t.cdf(i, n-k-1), stats.t.sf(i, n-k-1)) for i in t_stat] # holds p-values for each coefficient
        return {
            "R_squared": R_squared,
            "RMSE": RMSE,
            "F_stat": F_stat,
            "F_pvalue": F_pvalue,
            "t_stat": t_stat,
            "p_values": p_values
        }