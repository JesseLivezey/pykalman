'''
=============
Kalman Module
=============

This module provides inference methods for state-space estimation in continuous
spaces.
'''

from .standard import KalmanFilter
from .batched_standard import BatchedKalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter
from .kalman_classifier import KalmanFilterClassifier

__all__ = [
    "KalmanFilter",
    "BatchedKalmanFilter",
    "KalmanFilterClassifier",
    "AdditiveUnscentedKalmanFilter",
    "UnscentedKalmanFilter",
    "datasets",
    "sqrt"
]
