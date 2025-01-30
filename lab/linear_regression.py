import numpy as np
import scipy.stats as stats


class LinearRegression:

    def __init__(self):
        self._d = None
        self._n = None
        self._con_lvl = 0.95
        self.b = None

    @property
    def d(self):
        return self._d
    
    @property
    def n(self):
        return self._n

    @property
    def confidence_level(self):
        return self._con_lvl

    def fit(self, X, y):
        self.b = np.linalg.pinv(X.T @ X) @ X.T @ y
        self._d = len(self.b) - 1
        self._n = y.shape[0]

    def predict(self, X):
        return X @ self.b

    def variance(self, X, y):
        SSE = np.sum(np.square(y - X @ self.b))
        return SSE / (self._n - self._d - 1)

    def standard_deviation(self, X, y):
        var = self.variance(X, y)
        return np.sqrt(var)

    def significance(self, X, y):
        var = self.variance(X, y)
        std_dev = np.sqrt(var)
        SSE = np.sum(np.square(y - X @ self.b))
        SST = np.sum(np.square(y - np.mean(y)))
        SSR = SST - SSE
        f_stat = (SSR / self._d) / var
        f_pvalue = stats.f.sf(f_stat, self._d, self._n - self._d - 1)
        cov_matrix = np.linalg.pinv(X.T @ X) * var
        ti_stat = [self.b[i] / (std_dev * np.sqrt(cov_matrix[i, i])) for i in range(self._d)]
        ti_pvalues = [2 * min(stats.t.cdf(i, self._n - self._d - 1), stats.t.sf(i, self._n - self._d - 1)) for i in ti_stat]
        return {
            "f_pvalue": f_pvalue, 
            "ti_pvalues": ti_pvalues
        }

    def relevance(self, X, y):
        SSE = np.sum(np.square(y - X @ self.b))
        SST = np.sum(np.square(y - np.mean(y)))
        SSR = SST - SSE
        R_squared = SSR / SST
        return R_squared
    
    def test_relevance(self, X, y):
        SSE = np.sum(np.square(y - X @ self.b))
        RSE = np.sqrt((1 / (self._n - 2)) * SSE)
        MSE = (1 / self._n) * SSE
        RMSE = np.sqrt(MSE)
        return {
            "RSE": RSE,
            "MSE": MSE, 
            "RMSE": RMSE
        }

    def pearson(self, X, y):
        kin_geo = stats.pearsonr(X[:,1], X[:,2])
        kin_ine = stats.pearsonr(X[:,1], X[:,3])
        geo_ine = stats.pearsonr(X[:,2], X[:,3])
        return {
            "kin_geo": kin_geo,
            "kin_ine": kin_ine,
            "geo_ine": geo_ine
        }