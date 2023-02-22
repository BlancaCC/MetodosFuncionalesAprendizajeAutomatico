from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance
from scipy import linalg


def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X @ X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    l: float
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkovski', p=1.0)
    return A * np.exp(- d / l)


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """
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
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def kernel_pca(
    X: np.ndarray,
    X_test: np.ndarray,
    kernel: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    kernel: 
        function

    Returns
    -------
    X_test_hat:
        Projection of X_test on the principal components
    lambda_eigenvals:
        Eigenvalues of the centered kernel
    alpha_eigenvecs:
        Principal components. These are the eigenvectors
        of the centered kernel with the RKHS normalization

    Notes
    -------
    In the corresponding method of sklearn the eigenvectors
    are normalized in l2.

    """

    # In order to compute kernel PCA the following 6 steps are needed: 

    # 1. Build the Gram matrix K for the set of observations X 
    gram_matrix = kernel(X,X) # For n = size(X) then K is a n x n matrix 

    # 2. Compute the Gram matrix of the centered kernel
    k_len, _ =  X.shape 
    ones_matrix = np.ones((k_len, k_len))
    tilda_K = ( gram_matrix
                - 1/k_len * gram_matrix @ ones_matrix
                -  1/k_len * ones_matrix @ gram_matrix
                + 1 /(k_len**2) * ones_matrix @ gram_matrix @ ones_matrix
    )
    # 3.1 Find the eigenvalues: 
    # Since tilda_K is symmetric and positive semi-definite we are going to use 
    # eigh function from: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
    # The advantage is that it gives sorted the eigenvalues
    # All are going to be real numbers and we are going to take non zero vectors. 
    eigenvalues, normalized_eigenvector = linalg.eigh(tilda_K)
    eigenvalues_eigenvector = [(v,np.array(u)) 
               for v,u in  zip(eigenvalues,zip(*normalized_eigenvector)) 
               if v > 10**(-5) # zero tolerance
            ]
    eigenvalues_eigenvector = eigenvalues_eigenvector[::-1] # more relevan first
    
    # 3.2 Normalization condition 
    # eigenvectors should verify: alpha.T alpha = 1/lambda 
    # (where alpha it the eugenvector of  lambda)
    alpha_eigenvecs = [u / np.sqrt(v) for v,u in eigenvalues_eigenvector]
    lambda_eigenvals = list(map(lambda x:x[0],eigenvalues_eigenvector))

    # 4. Compute the matrix the test centered gram matrix 
    test_len, _ = X_test.shape
    test_gram_matrix = kernel(X_test, X)
    # centered
    tilda_ones = np.ones((test_len, k_len))
    test_centered_gram_matrix = (
        # TODO
    )


    return X_test_hat, lambda_eigenvals, alpha_eigenvecs


if __name__ == '__main__':
    X = np.array([
        [1,0,0],
        [0,2,0],
        [0,0,3]
    ])

    kernel = linear_kernel
    kernel_pca(X, X, kernel)