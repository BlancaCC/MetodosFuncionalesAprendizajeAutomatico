# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""
# Load packages

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """Vectorized RBF kernel (covariance) function.

    Parameters
    ----------
    X:
        Data matrix
    X_prime:
        Data matrix
    A:
        Output variance
    ls:
        Kernel lengthscale

    Returns
    -------
    kernel matrix

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_processes as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    [[3.         2.88236832]
     [2.88236832 3.        ]
     [2.55643137 2.88236832]]

    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def simulate_gp(
    t: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    M: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Gaussian process.

        X(t) ~ GP(mean_fn,kernel_fn)

    Parameters
    ----------
    t :
        Times at which the process is monitored.

    mean_fn:
        Mean function of the Gaussian process (vectorized).

    kernel_fn:
        Covariance functions of the Gaussian process (vectorized).

    M :
        Number of trajectories that are simulated.

    Returns
    -------
    X:
        Simulated trajectories as an np.ndarray with M rows and len(t) columns.
        Each trajectory is a row of the matrix consisting of the
        values of the process for each value of t.

    mean_vector:
        Vector with the values of the mean for each value of t.
        It is a np.ndarray with len(t) columns.

    kernel_matrix:
        Kernel matrix as an np.ndarray with len(t) rows and len(t)  columns.


    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_processes as gp
    >>> def mean_fn(t):
    ...     return np.zeros(np.shape(t))
    >>> def BB_kernel(s,t):
    ...     return (np.minimum(s,t) - s * t)
    >>> M, N  = (20, 1000)
    >>> t0, t1 = (0.0, 1.0)
    >>> t = np.linspace(t0, t1, N)
    >>> BB, _, _ = gp.simulate_gp(t, mean_fn, BB_kernel, M)
    >>> _ = plt.plot(t, BB.T)
    >>> _= plt.xlabel('t')
    >>> _=  plt.ylabel('BB(t)')
    >>> _= plt.title('Standard Brownian Bridge process')
    >>> plt.show()
    """
    #  NOTE Use np.meshgrid for the arguments of
    #  kernel_fn to compute the kernel matrix.
    #  Do not use numpy.random.multivariate_normal
    #  Use np.linalg.svd
    #

    # Gaussian processes definitions
    mean_vector = mean_fn(t) # Shape: N x 1
    # Sources: 
    # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    # https://interactivechaos.com/es/manual/tutorial-de-numpy/la-funcion-meshgrid
    T_1, T_2 = np.meshgrid(t, t, 
                             indexing='ij' # matrix indexing
                            )
    kernel_matrix = kernel_fn(T_1, T_2)# N x N 
    # we are going to carry Cholesky decomposition K(t,t) = LL^t
    # To compute L: 
    # Using svd decomposition: K = u s v^t
    # Since K is symmetric then u = v then K = (u sqrt(s))(u sqrt(s))^t
    # So L = u sqrt(s)
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

    u, s, _ = np.linalg.svd(kernel_matrix)
    S = np.diag(s)
    L = u @ np.sqrt(S)

    # Simulation X(t) = m(t) + LZ with Z follows a N(0, Id_n)
    Z = np.random.randn(M, len(t))
    X = mean_vector + Z @ L

    return X, mean_vector, kernel_matrix


def simulate_conditional_gp(
    t: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    M: int,
) -> np.ndarray:
    """Simulate a Gaussian process conditined to observed values.

        X(t) ~ GP(mean_fn,kernel_fn)

        condition to having observed  X(t_obs) = x_obs at t_obs

    Parameters
    ----------
    t :
        Times at which the process is monitored.

    t_obs :
        Times at which the values of the process have been observed.
        The Gaussian process has the value x_obs at t_obs.

    x_obs :
        Values of the process at t_obs.

    mean_fn :
        Mean function of the Gaussian process [vectorized].

    kernel_fn :
        Covariance functions of the Gaussian process.

    M :
        Number of trajectories in the simulation.

    Returns
    -------
    X:
        Simulated trajectories as an np.ndarray with M rows and len(t) columns.
        Each trajectory is a row of the matrix consisting of the
        values of the process for each value of t.

    mean_vector:
        Vector with the values of the mean for each value of t.
        It is a np.ndarray with len(t) columns.

    kernel_matrix:
        Kernel matrix as an np.ndarray with len(t) rows and len(t)  columns.

    Example
    -------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_processes as gp
    >>> def mean_fn(t, mu=1.0):
    ...     return mu*t
    >>> def BB_kernel(s,t):
    ...     return np.minimum(s,t) - s * t
    >>> M, N  = (30, 1000)
    >>> t0, t1 = (0.0, 1.0)
    >>> t = np.linspace(t0, t1, N)
    >>> t_obs = np.array([0.25, 0.5, 0.75])
    >>> x_obs = np.array([0.3, -0.3, -1.0])
    >>> B, _, _ = gp.simulate_conditional_gp(
    ...     t,
    ...     t_obs,
    ...     x_obs,
    ...     mean_fn,
    ...     BB_kernel,
    ...     M,
    ... )
    >>> _ = plt.plot(t, B.T)
    >>> _ = plt.xlabel('t')
    >>> _ =  plt.ylabel('B(t)')

    """
    # NOTE Use 'multivariate_normal' from numpy with "'method = 'svd'".
    # 'svd' is slower, but numerically more robust than 'cholesky'

    #<YOUR CODE HERE>

    T_1, T_2 = np.meshgrid(t, t, indexing='ij')
    T_1_obs, T_2_obs = np.meshgrid(t_obs, t_obs, indexing='ij')
    T_1_t_obs, T_2_t_obs = np.meshgrid(t, t_obs, indexing='ij')
    T_1_obs_t, T_2_obs_t = np.meshgrid(t_obs, t, indexing='ij')
    
    kernel_matrix_t = kernel_fn(T_1, T_2)
    kernel_matrix_obs = kernel_fn(T_1_obs, T_2_obs)
    kernel_matrix_t_obs = kernel_fn(T_1_t_obs, T_2_t_obs)
    kernel_matrix_obs_t = kernel_fn(T_1_obs_t, T_2_obs_t)


    inverse_kernel = np.linalg.inv(kernel_matrix_obs)

    mean_vector = mean_fn(t) + np.dot(np.dot(kernel_matrix_t_obs,inverse_kernel), x_obs - mean_fn(t_obs))
    kernel_matrix = kernel_matrix_t - np.dot(np.dot(kernel_matrix_t_obs,inverse_kernel), kernel_matrix_obs_t) 

    print(np.array(mean_vector))
    print(np.array(kernel_matrix))

    X = np.random.default_rng().multivariate_normal(mean_vector, kernel_matrix, M,  method = 'svd')

    return X, mean_vector, kernel_matrix


def gp_regression(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sigma2_noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Gaussian process regression.

    Parameters
    ----------
    X:
        :math:`N \times D` data matrix for training

    y:
        vector of output values

    X_test:
        :math:`L \times D` data matrix for testing.

    kernel_fn:
        Kernel (covariance) function.

    sigma2_noise:
        Variance of the noise.
        It is a hyperparameter of GP regression.

    Returns
    -------
        prediction_mean:
            Predictions at the test points.

        prediction_variance:
            Uncertainty of the predictions.
    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_processes as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> y = [1, 2, 3]
    >>> X_test = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> sigma2_noise = 0.01
    >>> def kernel (X, X_prime):
    ...     return gp.rbf_kernel(X, X_prime, A, l)
    >>> predictions, _ = gp.gp_regression(X, y, X_test, kernel, sigma2_noise)
    >>> print(predictions)
    [1.00366515 2.02856104]
    """

    # NOTE use 'np.linalg.solve' instead of inverting the matrix.
    # This procedure is numerically more robust.

    #<YOUR CODE HERE>
    return 0
    #return prediction_mean, prediction_variance


if __name__ == "__main__":
    import doctest
    doctest.testmod()
