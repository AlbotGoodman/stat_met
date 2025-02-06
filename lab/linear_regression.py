import numpy as np
import scipy.stats as stats


class LinearRegression:

    def __init__(self):
        self._d = None
        self._n = None
        self._con_lvl = 0.95
        self._b = None

    @property
    def d(self):
        return self._d
    
    @property
    def n(self):
        return self._n

    @property
    def confidence_level(self):
        return self._con_lvl
    
    @property
    def alpha(self):
        return 1 - self._con_lvl
    
    def fit(self, X, y):
        self._b = np.linalg.pinv(X.T @ X) @ X.T @ y
        self._d = len(self._b) - 1
        self._n = y.shape[0]

    def predict(self, X):                                               # val
        return X @ self._b

    def _errors(self, X, y):
        return y - self.predict(X)

    def _sse(self, X, y):
        return np.sum(np.square(self._errors(X, y)))
    
    def _sst(self, y):
        return np.sum(np.square(y - np.mean(y)))
    
    def _ssr(self, X, y):
        return self._sst(y) - self._sse(X, y)

    def _variance(self, X, y):
        return self._sse(X, y) / (self._n - self._d - 1)

    def _standard_deviation(self, X, y):
        return np.sqrt(self._variance(X, y))
    
    def _cov_matrix(self, X, y):
        return np.linalg.pinv(X.T @ X) * self._variance(X, y)

    def significance(self, X, y):
        f_stat = (self._ssr(X, y) / self._d) / self._variance(X, y)
        f_pvalue = stats.f.sf(f_stat, self._d, self._n - self._d - 1)
        ti_stat = [self._b[i] / (self._standard_deviation(X, y) * np.sqrt(self._cov_matrix(X, y)[i, i])) for i in range(self._d)]
        ti_pvalues = [2 * min(stats.t.cdf(i, self._n - self._d - 1), stats.t.sf(i, self._n - self._d - 1)) for i in ti_stat]
        return {
            "f_pvalue": f_pvalue, 
            "ti_pvalues": ti_pvalues
        }

    def r_squared(self, X, y):                                          # val
        return self._ssr(X, y) / self._sst(y)
    
    def relevance(self, X, y):                                          # val
        RSE = np.sqrt(self._sse(X, y) / (self._n - 2))
        MSE = (1 / self._n) * self._sse(X, y)
        RMSE = np.sqrt(MSE)
        return {
            "RSE": RSE,
            "MSE": MSE, 
            "RMSE": RMSE
        }

    def pearson(self, X):
        r_X = X[:, 1:] # Remove the intercept column to avoid division by zero
        return np.corrcoef(r_X, rowvar=False)
    
    def confidence_intervals(self, X, y):
        t_crit = stats.t.ppf(1 - self.alpha / 2, self._n - self._d - 1)
        ci = []
        for i in range(self._d + 1):
            margin = t_crit * self._standard_deviation(X, y) * np.sqrt(self._cov_matrix(X, y)[i,i])
            ci.append(margin)
        return ci