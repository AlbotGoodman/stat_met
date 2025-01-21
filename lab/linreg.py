import numpy as np
import scipy.stats as stats

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.p_values_ = None
        self.r_squared_ = None
        self.adj_r_squared_ = None

    def fit(self, X, y):
        n = X.shape[0]
        p = X.shape[1]
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        y_hat = X @ beta
        residuals = y - y_hat
        sse = residuals.T @ residuals
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]
        self.r_squared_ = 1 - sse / ((n - 1) * np.var(y))
        self.adj_r_squared_ = 1 - (1 - self.r_squared_) * (n - 1) / (n - p - 1)
        self.p_values_ = np.array([2 * (1 - stats.t.cdf(np.abs(beta[i] / np.sqrt(np.diag(np.linalg.inv(X.T @ X))[i, i])), n - p - 1)) for i in range(p + 1)])

    def predict(self, X):
        return self.intercept_ + X @ self.coef_

    def summary(self):
        print('Coefficients:')
        print('Intercept:', self.intercept_)
        for i in range(len(self.coef_)):
            print(f'X{i + 1}:', self.coef_[i])
        print('P-values:', self.p_values_)
        print('R-squared:', self.r_squared_)
        print('Adjusted R-squared:', self.adj_r_squared_)