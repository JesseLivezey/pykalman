"""
=====================================
Inference for Linear-Gaussian Systems
=====================================

This module implements the Kalman Filter, Kalman Smoother, and
EM Algorithm for Linear-Gaussian state space models
"""
import warnings

import numpy as np
from scipy import linalg

from .standard import (KalmanFilter, _determine_dimensionality, _filter,
                       _smooth, _smooth_pair, _em)
from .utils import array1d, array2d, check_random_state, \
    get_params, log_multivariate_normal_density, preprocess_arguments


# Dimensionality of each Kalman Filter parameter for a single time step
DIM = {
    'transition_matrices': 2,
    'transition_offsets': 1,
    'observation_matrices': 2,
    'observation_offsets': 1,
    'transition_covariance': 2,
    'observation_covariance': 2,
    'initial_state_mean': 1,
    'initial_state_covariance': 2,
}


class BatchedKalmanFilter(KalmanFilter):
    """Implements the Kalman Filter, Kalman Smoother, and EM algorithm running
    on batched data.

    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
    for a Linear Gaussian model specified by,

    .. math::

        x_{t+1}   &= A_{t} x_{t} + b_{t} + \\text{Normal}(0, Q_{t}) \\\\
        z_{t}     &= C_{t} x_{t} + d_{t} + \\text{Normal}(0, R_{t})

    The Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | z_{0:t})`.  As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `filtered_state_means[t]` and covariances `filtered_state_covariances[t]`.

    Similarly, the Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | z_{0:T-1})`.

    The EM algorithm aims to find for
    :math:`\\theta = (A, b, C, d, Q, R, \\mu_0, \\Sigma_0)`

    .. math::

        \\max_{\\theta} P(z_{0:T-1}; \\theta)

    If we define :math:`L(x_{0:T-1},\\theta) = \\log P(z_{0:T-1}, x_{0:T-1};
    \\theta)`, then the EM algorithm works by iteratively finding,

    .. math::

        P(x_{0:T-1} | z_{0:T-1}, \\theta_i)

    then by maximizing,

    .. math::

        \\theta_{i+1} = \\arg\\max_{\\theta}
            \\mathbb{E}_{x_{0:T-1}} [
                L(x_{0:T-1}, \\theta)| z_{0:T-1}, \\theta_i
            ]

    Parameters
    ----------
    transition_matrices : [n_timesteps-1, n_dim_state, n_dim_state] or \
    [n_dim_state,n_dim_state] array-like
        Also known as :math:`A`.  state transition matrix between times t and
        t+1 for t in [0...n_timesteps-2]
    observation_matrices : [n_timesteps, n_dim_obs, n_dim_state] or [n_dim_obs, \
    n_dim_state] array-like
        Also known as :math:`C`.  observation matrix for times
        [0...n_timesteps-1]
    transition_covariance : [n_dim_state, n_dim_state] array-like
        Also known as :math:`Q`.  state transition covariance matrix for times
        [0...n_timesteps-2]
    observation_covariance : [n_dim_obs, n_dim_obs] array-like
        Also known as :math:`R`.  observation covariance matrix for times
        [0...n_timesteps-1]
    transition_offsets : [n_timesteps-1, n_dim_state] or [n_dim_state] \
    array-like
        Also known as :math:`b`.  state offsets for times [0...n_timesteps-2]
    observation_offsets : [n_timesteps, n_dim_obs] or [n_dim_obs] array-like
        Also known as :math:`d`.  observation offset for times
        [0...n_timesteps-1]
    initial_state_mean : [n_dim_state] array-like
        Also known as :math:`\\mu_0`. mean of initial state distribution
    initial_state_covariance : [n_dim_state, n_dim_state] array-like
        Also known as :math:`\\Sigma_0`.  covariance of initial state
        distribution
    random_state : optional, numpy random state
        random number generator used in sampling
    em_vars : optional, subset of ['transition_matrices', \
    'observation_matrices', 'transition_offsets', 'observation_offsets', \
    'transition_covariance', 'observation_covariance', 'initial_state_mean', \
    'initial_state_covariance'] or 'all'
        if `em_vars` is an iterable of strings only variables in `em_vars`
        will be estimated using EM.  if `em_vars` == 'all', then all
        variables will be estimated.
    n_dim_state: optional, integer
        the dimensionality of the state space. Only meaningful when you do not
        specify initial values for `transition_matrices`, `transition_offsets`,
        `transition_covariance`, `initial_state_mean`, or
        `initial_state_covariance`.
    n_dim_obs: optional, integer
        the dimensionality of the observation space. Only meaningful when you
        do not specify initial values for `observation_matrices`,
        `observation_offsets`, or `observation_covariance`.
    """
    def __init__(self, transition_matrices=None, observation_matrices=None,
            transition_covariance=None, observation_covariance=None,
            transition_offsets=None, observation_offsets=None,
            initial_state_mean=None, initial_state_covariance=None,
            random_state=None,
            em_vars=['transition_covariance', 'observation_covariance',
                     'initial_state_mean', 'initial_state_covariance'],
            n_dim_state=None, n_dim_obs=None):
        """Initialize Kalman Filter"""

        # determine size of state space
        n_dim_state = _determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (transition_offsets, array1d, -1),
             (transition_covariance, array2d, -2),
             (initial_state_mean, array1d, -1),
             (initial_state_covariance, array2d, -2),
             (observation_matrices, array2d, -1)],
            n_dim_state
        )
        n_dim_obs = _determine_dimensionality(
            [(observation_matrices, array2d, -2),
             (observation_offsets, array1d, -1),
             (observation_covariance, array2d, -2)],
            n_dim_obs
        )

        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.random_state = random_state
        self.em_vars = em_vars
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs

    def batched_sample(self, n_timesteps, initial_state=None, random_state=None):
        """Sample a state sequence :math:`n_{\\text{timesteps}}` timesteps in
        length.

        Parameters
        ----------
        n_batch : int
            number of timesteps
        n_timesteps : int
            number of timesteps

        Returns
        -------
        states : [n_batch, n_timesteps, n_dim_state] array
            hidden states corresponding to times [0...n_timesteps-1]
        observations : [n_batch, n_timesteps, n_dim_obs] array
            observations corresponding to times [0...n_timesteps-1]
        """
        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        n_dim_state = transition_matrices.shape[-2]
        n_dim_obs = observation_matrices.shape[-2]
        states = np.ma.zeros((n_batch, n_timesteps, n_dim_state))
        observations = np.ma.zeros((n_batch, n_timesteps, n_dim_obs))

        for ii in range(n_batch):
            st, ob = self.sample(n_timesteps, initial_state, random_state)
            states[ii] = st
            observations[ii] = ob

        return (states, np.ma.array(observations))

    def batched_filter(self, X):
        """Apply the Kalman Filter

        Apply the Kalman Filter to estimate the hidden state at time :math:`t`
        for :math:`t = [0...n_{\\text{timesteps}}-1]` given observations up to
        and including time `t`.  Observations are assumed to correspond to
        times :math:`[0...n_{\\text{timesteps}}-1]`.  The output of this method
        corresponding to time :math:`n_{\\text{timesteps}}-1` can be used in
        :func:`KalmanFilter.filter_update` for online updating.

        Parameters
        ----------
        X : [n_batch, n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]` is masked, then `X[t]` will be
            treated as a missing observation.

        Returns
        -------
        filtered_state_means : [n_batch, n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1]
            given observations up to and including the current time step
        filtered_state_covariances : [n_batch, n_timesteps, n_dim_state, n_dim_state] \
        array
            covariance matrix of hidden state distributions for times
            [0...n_timesteps-1] given observations up to and including the
            current time step
        """
        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
         )
        n_batch, n_timesteps, n_dim_obs = X.shape
        n_dim_state = initial_state_mean.shape[0]
        filtered_state_means = np.ma.empty((n_batch, n_timesteps, n_dim_state))
        filtered_state_covariances = np.ma.empty((n_batch, n_timesteps,
                                                  n_dim_state, n_dim_state))
        for ii in range(n_batch):
            mns, cvs = self.filter(X[ii])
            filtered_state_means[ii] = mns
            filtered_state_covariances[ii] = cvs

        return (filtered_state_means, filtered_state_covariances)

    def batched_filter_update(self, filtered_state_mean, filtered_state_covariance,
                      observation=None, transition_matrix=None,
                      transition_offset=None, transition_covariance=None,
                      observation_matrix=None, observation_offset=None,
                      observation_covariance=None):
        r"""Update a Kalman Filter state estimate

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Parameters
        ----------
        filtered_state_mean : [n_batch, n_dim_state] array
            mean estimate for state at time t given observations from times
            [1...t]
        filtered_state_covariance : [n_batch, n_dim_state, n_dim_state] array
            covariance of estimate for state at time t given observations from
            times [1...t]
        observation : [n_batch, n_dim_obs] array or None
            observation from time t+1.  If `observation` is a masked array and
            any of `observation`'s components are masked or if `observation` is
            None, then `observation` will be treated as a missing observation.
        transition_matrix : optional, [n_dim_state, n_dim_state] array
            state transition matrix from time t to t+1.  If unspecified,
            `self.transition_matrices` will be used.
        transition_offset : optional, [n_dim_state] array
            state offset for transition from time t to t+1.  If unspecified,
            `self.transition_offset` will be used.
        transition_covariance : optional, [n_dim_state, n_dim_state] array
            state transition covariance from time t to t+1.  If unspecified,
            `self.transition_covariance` will be used.
        observation_matrix : optional, [n_dim_obs, n_dim_state] array
            observation matrix at time t+1.  If unspecified,
            `self.observation_matrices` will be used.
        observation_offset : optional, [n_dim_obs] array
            observation offset at time t+1.  If unspecified,
            `self.observation_offset` will be used.
        observation_covariance : optional, [n_dim_obs, n_dim_obs] array
            observation covariance at time t+1.  If unspecified,
            `self.observation_covariance` will be used.

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t+1 given observations from times
            [1...t+1]
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t+1 given observations
            from times [1...t+1]
        """
        raise NotImplementedError
        # initialize matrices
        (transition_matrices, transition_offsets, transition_cov,
         observation_matrices, observation_offsets, observation_cov,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )
        transition_offset = _arg_or_default(
            transition_offset, transition_offsets,
            1, "transition_offset"
        )
        observation_offset = _arg_or_default(
            observation_offset, observation_offsets,
            1, "observation_offset"
        )
        transition_matrix = _arg_or_default(
            transition_matrix, transition_matrices,
            2, "transition_matrix"
        )
        observation_matrix = _arg_or_default(
            observation_matrix, observation_matrices,
            2, "observation_matrix"
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov,
            2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov,
            2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        n_batch, n_dim_state = filtered_state_mean.shape[1]
        next_filtered_state_mean = np.ma.empty((n_batch, n_dim_state))
        next_filtered_state_covariance = np.ma.empty((n_batch, n_dim_state, n_dim_state))

        for ii in range(n_batch):
            nfsm, nfsc = filter_update(filtered_state_mean[ii],
                                       filtered_state_covariance[ii],
                                       observation, transition_matrix,
                                       transition_offset, transition_covariance,
                                       observation_matrix, observation_offset,
                                       observation_covariance)
            next_filtered_state_mean[ii] = nfsm
            next_filtered_state_covariance[ii] = nfsc


        return (next_filtered_state_mean, next_filtered_state_covariance)

    def batched_smooth(self, X):
        """Apply the Kalman Smoother

        Apply the Kalman Smoother to estimate the hidden state at time
        :math:`t` for :math:`t = [0...n_{\\text{timesteps}}-1]` given all
        observations.  See :func:`_smooth` for more complex output

        Parameters
        ----------
        X : [n_batch, n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]` is masked, then `X[t]` will be
            treated as a missing observation.

        Returns
        -------
        smoothed_state_means : [n_batch, n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1]
            given all observations
        smoothed_state_covariances : [n_batch, n_timesteps, n_dim_state]
            covariances of hidden state distributions for times
            [0...n_timesteps-1] given all observations
        """
        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
         )
        n_batch, n_timesteps, n_dim_obs = X.shape
        n_dim_state = initial_state_mean.shape[0]
        smoothed_state_means = np.ma.empty((n_batch, n_timesteps, n_dim_state))
        smoothed_state_covariances = np.ma.empty((n_batch, n_timesteps,
                                                  n_dim_state, n_dim_state))
        for ii in range(n_batch):
            mns, cvs = self.smooth(X[ii])
            smoothed_state_means[ii] = mns
            smoothed_state_covariances[ii] = cvs

        return (smoothed_state_means, smoothed_state_covariances)

    def batched_em(self, X, y=None, n_iter=10, em_vars=None):
        """Apply the EM algorithm

        Apply the EM algorithm to estimate all parameters specified by
        `em_vars`.  Note that all variables estimated are assumed to be
        constant for all time.  See :func:`_em` for details.

        Parameters
        ----------
        X : [n_batch, n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]`'s components is masked, then
            `X[t]` will be treated as a missing observation.
        n_iter : int, optional
            number of EM iterations to perform
        em_vars : iterable of strings or 'all'
            variables to perform EM over.  Any variable not appearing here is
            left untouched.
        """
        assert X.ndim == 3
        n_batch = X.shape[0]

        # initialize parameters
        (self.transition_matrices, self.transition_offsets,
         self.transition_covariance, self.observation_matrices,
         self.observation_offsets, self.observation_covariance,
         self.initial_state_mean, self.initial_state_covariance) = (
            self._initialize_parameters()
        )

        # Create dictionary of variables not to perform EM on
        if em_vars is None:
            em_vars = self.em_vars

        if em_vars == 'all':
            given = {}
        else:
            given = {
                'transition_matrices': self.transition_matrices,
                'observation_matrices': self.observation_matrices,
                'transition_offsets': self.transition_offsets,
                'observation_offsets': self.observation_offsets,
                'transition_covariance': self.transition_covariance,
                'observation_covariance': self.observation_covariance,
                'initial_state_mean': self.initial_state_mean,
                'initial_state_covariance': self.initial_state_covariance
            }
            em_vars = set(em_vars)
            for k in list(given.keys()):
                if k in em_vars:
                    given.pop(k)

        # If a parameter is time varying, print a warning
        for (k, v) in get_params(self).items():
            if k in DIM and (not k in given) and len(v.shape) != DIM[k]:
                warn_str = (
                    '{0} has {1} dimensions now; after fitting, '
                    + 'it will have dimension {2}'
                ).format(k, len(v.shape), DIM[k])
                warnings.warn(warn_str)

        # Actual EM iterations
        for ii in range(n_iter):
            tm = np.zeros_like(self.transition_matrices)
            om = np.zeros_like(self.observation_matrices)
            to = np.zeros_like(self.transition_offsets)
            oo = np.zeros_like(self.observation_offsets)
            tc = np.zeros_like(self.transition_covariance)
            oc = np.zeros_like(self.observation_covariance)
            ism = np.zeros_like(self.initial_state_mean)
            isc = np.zeros_like(self.initial_state_covariance)
            for jj in range(n_batch):
                Z = self._parse_observations(X[jj])
                (predicted_state_means, predicted_state_covariances,
                 kalman_gains, filtered_state_means,
                 filtered_state_covariances) = (
                    _filter(
                        self.transition_matrices, self.observation_matrices,
                        self.transition_covariance, self.observation_covariance,
                        self.transition_offsets, self.observation_offsets,
                        self.initial_state_mean, self.initial_state_covariance,
                        Z
                    )
                )
                (smoothed_state_means, smoothed_state_covariances,
                 kalman_smoothing_gains) = (
                    _smooth(
                        self.transition_matrices, filtered_state_means,
                        filtered_state_covariances, predicted_state_means,
                        predicted_state_covariances
                    )
                )
                sigma_pair_smooth = _smooth_pair(
                    smoothed_state_covariances,
                    kalman_smoothing_gains
                )
                (tmj, omj, toj, ooj, tcj, ocj, ismj, iscj) = (
                    _em(Z, self.transition_offsets, self.observation_offsets,
                        smoothed_state_means, smoothed_state_covariances,
                        sigma_pair_smooth, given=given
                    )
                )
                for cm, it in zip((tm, om, to, oo, tc, oc, ism, isc),
                                   (tmj, omj, toj, ooj, tcj, ocj, ismj, iscj)):
                    cm[:] = cm + it
            for cm in (tm, om, to, oo, tc, oc, ism, isc):
                cm[:] = cm/n_batch

            (self.transition_matrices,  self.observation_matrices,
             self.transition_offsets, self.observation_offsets,
             self.transition_covariance, self.observation_covariance,
             self.initial_state_mean, self.initial_state_covariance) = (
                     tm, om, to, oo, tc, oc, ism, isc)


        return self

    def batched_loglikelihood(self, X):
        """Calculate the log likelihood of all observations

        Parameters
        ----------
        X : [n_batch, n_timesteps, n_dim_obs] array
            observations for time steps [0...n_timesteps-1]

        Returns
        -------
        likelihood : float
            likelihood of all observations
        """
        n_batch = X.shape[0]
        loglikelihoods = np.empty(n_batch)
        for ii in range(n_batch):
            loglikelihoods[ii] = self.loglikelihood(X[ii])

        return loglikelihoods

    def mean_loglikelihood(self, X):
        """Calculate the log likelihood of all observations

        Parameters
        ----------
        X : [n_batch, n_timesteps, n_dim_obs] array
            observations for time steps [0...n_timesteps-1]

        Returns
        -------
        likelihood : float
            likelihood of all observations
        """
        return self.batched_loglikelihood(X).mean()
