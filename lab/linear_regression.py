import numpy as np
import scipy.stats as stats


class LinearRegression:

    def __init__(self):
        self._d = None
        self._n = None
        self._con_lvl = 0.95
        self._b = None
        self._X_train_means = None
        self._X_train_std_devs = None
        self._y_train_mean = None
        self._y_train_std_dev = None

    @property
    def d(self):
        """Returns the number of parameters/features."""
        return self._d
    
    @property
    def n(self):
        """Returns the number of rows in the training set."""
        return self._n
    
    @property
    def n_sample(self):
        """Returns the number of rows of the sample set (validation/test)."""
        return self._n_sample

    @property
    def confidence_level(self):
        """Returns the confidence level."""
        return self._con_lvl
    
    @property
    def alpha(self):
        """Returns the alpha value which is based on 1 - confidence level."""
        return 1 - self._con_lvl
    
    def standardise(arr, is_X=True):
        """
        Returning a standardised numpy array. 

        Args: 
            arr:    the data set, as a numpy array
            is_X:   if set to True, the feature matrix should be standardised (only the continuous values)
                    if set to False, the independent variable should be standardised
        """
        data_std = arr.copy()
        
        if is_X:
            d = data_std.shape[1]
            for i in range(1, d-1):
                mean = np.mean(data_std[:, i])
                std_dev = np.std(data_std[:, i], ddof=1) # since this is not the population
                data_std[:, i] = (data_std[:, i] - mean) / std_dev

        else:
            mean = np.mean(data_std)
            std_dev = np.std(data_std, ddof=1)
            data_std = (data_std - mean) / std_dev

        return data_std
    
    def fit(self, X, y):
        """Trains the regression model."""
        self._b = np.linalg.pinv(X.T @ X) @ X.T @ y
        self._d = len(self._b) - 1
        self._n = y.shape[0]

    def predict(self, X):
        """Calculates and returns the predicted dependent variables (ŷ)."""
        y_hat = X @ self._b
        self._n_sample = y_hat.shape[0]
        return y_hat

    def _errors(self, X, y):
        """Helper method that returns the errors (y - ŷ)."""
        return y - self.predict(X)

    def _sse(self, X, y):
        """Helper method that returns the sum of squared errors."""
        return np.sum(np.square(self._errors(X, y)))
    
    def _sst(self, y):
        """Helper method that returns the total sum of squares."""
        return np.sum(np.square(y - np.mean(y)))
    
    def _ssr(self, X, y):
        """Helper method that returns the sum of squares regression."""
        return self._sst(y) - self._sse(X, y)

    def _variance(self, X, y):
        """Helper method that returns the variance of the errors (σ²)."""
        return self._sse(X, y) / (self._n - self._d - 1)

    def _standard_deviation(self, X, y):
        """Helper method that areturns the standard deviation (σ)."""
        return np.sqrt(self._variance(X, y))
    
    def _cov_matrix(self, X, y):
        """Helper method that returns the covariance matrix."""
        return np.linalg.pinv(X.T @ X) * self._variance(X, y)

    def significance(self, X, y):
        """Tests significance of the whole regression model (F-statistic) and it's parameters (T-statistic)."""
        
        # F-statistic
        f_stat = (self._ssr(X, y) / self._d) / self._variance(X, y)
        f_pvalue = stats.f.sf(f_stat, self._d, self._n - self._d - 1)
        
        # T-statistic
        ti_stat = [self._b[i] / (self._standard_deviation(X, y) * np.sqrt(self._cov_matrix(X, y)[i, i])) for i in range(self._d)]
        ti_pvalues = [2 * min(stats.t.cdf(i, self._n - self._d - 1), stats.t.sf(i, self._n - self._d - 1)) for i in ti_stat]
        
        return {
            "f_pvalue": f_pvalue, 
            "ti_pvalues": ti_pvalues
        }

    def r_squared(self, X, y):
        """Returns the R² value of the regression model."""
        return self._ssr(X, y) / self._sst(y)
    
    def estimates(self, X, y):
        """
        Estimates the variation of the regression model.
        
        RSE: estimates the standard error
        MSE: estimates the variance
        RMSE: estimates the standard deviation
        """
        RSE = np.sqrt(self._sse(X, y) / (self._n_sample - self._d - 1))
        MSE = (1 / self._n_sample) * self._sse(X, y)
        RMSE = np.sqrt(MSE)
        return {
            "RSE": RSE,
            "MSE": MSE, 
            "RMSE": RMSE
        }

    def pearson(self, X):
        """Calculates the Pearson correlation between parameters and returns them in a matrix."""
        r_X = X[:, 1:] # Remove the intercept column to avoid division by zero
        return np.corrcoef(r_X, rowvar=False)
    
    def confidence_intervals(self, X, y):
        """Calculates the interval between which we will find [confidence level] % of the data. Returns the margin in a list."""
        t_crit = stats.t.ppf(1 - self.alpha / 2, self._n - self._d - 1)
        ci = []
        for i in range(self._d + 1):
            margin = t_crit * self._standard_deviation(X, y) * np.sqrt(self._cov_matrix(X, y)[i,i])
            ci.append(margin)
        return ci