"""
=====================================
Classification for Linear-Gaussian Systems
=====================================

"""
import warnings

import numpy as np

"""
from abc import ABCMeta
"""
from sklearn.base import ClassifierMixin

from batched_standard import BatchedKalmanFilter


class KalmanFilterClassifier(ClassifierMixin):
    """
    Does likelihood-based classification by fitting one model per class.
    """
    def __init__(self, n_classes, obs_dim, state_dim):
        self.classifiers = []
        for ii in range(n_classes):
            self.classifiers.append(BatchedKalmanFilter(n_dim_state=state_dim, n_dim_obs=obs_dim))

    def fit(self, X, y, n_iter=10):
        """
        Fit Kalman Filter models to labeled data.
        """
        for ii, cl in enumerate(self.classifiers):
            X_ii = X[y == ii]
            if X_ii.size > 0:
                cl.batched_em(X_ii, n_iter, em_vars='all')

    def predict_log_proba(self, X):
        lps = []
        for cl in self.classifiers:
            lps.append(cl.batched_loglikelihood(X)[:, np.newaxis])
        return np.hstack(lps)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).astype(int).mean()
