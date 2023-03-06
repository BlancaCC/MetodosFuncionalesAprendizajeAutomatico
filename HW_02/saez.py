
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Type
from sklearn.gaussian_process.kernels import RBF


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# needed for default implementation of set_params
from sklearn.base import BaseEstimator, TransformerMixin


class RandomFeaturesSampler(ABC, BaseEstimator, TransformerMixin):
    """Base class for random feature samplers."""

    def __init__(self, n_random_features: int) -> None:
        self.n_random_features = n_random_features
        self.w = None

    def _initialize_w(self, n_features: int) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Initialize w's for the random features.
        This should be implemented for each kernel."""
        pass

    def fit_transform(
        self, X: np.ndarray, X_prime: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Initialize  w's (fit) & compute random features (transform)."""
        n_features = np.shape(X)[1]
        self.fit(n_features)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute the random features.
        Assumes that the vector of w's has been initialized
        Parameters
        ----------
        X:
            Data matrix of shape (n_instances, n_features).
        Returns
        -------
        random_features:
            Array of shape (n_instances, n_random_features).
        """
        if self.w is None:
            raise ValueError("Use fit_transform to initialize w.")

        n_instances, n_features = np.shape(X)

        if np.shape(self.w)[1] != n_features:
            raise ValueError("Different # of features for X and w.")

        random_features = np.empty((n_instances, self.n_random_features))
        random_features[:, ::2] = np.cos(X @ self.w.T)
        random_features[:, 1::2] = np.sin(X @ self.w.T)

        norm_factor = np.sqrt(self.n_random_features // 2)
        random_features = random_features / norm_factor

        return random_features


class RandomFeaturesSamplerRBF(RandomFeaturesSampler):
    """Random Fourier Features for the RBF kernel."""

    def __init__(self, sigma: float = 1.0, n_random_features: int = 100) -> None:

        self.sigma = sigma
        # Initialize number of random features in parent class
        super().__init__(n_random_features)

    def fit(self, n_features: int = 100) -> np.ndarray:
        """Initialize the w's for the random features."""
        w_mean = np.zeros(n_features)
        w_cov_matrix = self.sigma**2 * np.identity(n_features)

        rng = np.random.default_rng()
        self.w = rng.multivariate_normal(
            w_mean,
            w_cov_matrix,
            self.n_random_features // 2,
        )


class RandomFeaturesSamplerMatern(RandomFeaturesSampler):
    """Random Fourier Features for the Matern kernel."""

    def __init__(
        self, scale: float = 1.0, nu: float = 1.0, n_random_features: int = 100
    ) -> None:
        """The Fourier transform of the Matérn kernel is a
        Student's t distribution with twice the degrees of freedom.
        Ref. Chapter 4 of
        Carl Edward Rasmussen and Christopher K. I. Williams. 2005.
        Gaussian Processes for Machine Learning
        (Adaptive Computation and Machine Learning). The MIT Press.
        There is probably a mistake with the scale factor.
        """

        # Changed variables since they CANNNOT be modified due to sklearn GridSearch
        # self.nu = 2.0 * nu
        self.nu = nu
        # Changed name from original's author name due to incompatibility with sklearn
        self.scale = scale

        super().__init__(n_random_features)
        self.w = None

    def fit(self, n_features: int) -> np.ndarray:
        """Compute w's for random Matérn features."""
        # Scale of the Fourier tranform of the kernel
        w_mean = np.zeros(n_features)
        w_cov_matrix = self.scale**2 * np.identity(n_features)

        self.w = random_multivariate_student_t(
            w_mean,
            w_cov_matrix,
            self.nu,
            self.n_random_features // 2,
        )


def random_multivariate_student_t(
    mean: np.ndarray,
    cov_matrix: np.ndarray,
    degrees_of_freedom: float,
    n_samples: int,
) -> np.ndarray:
    """Generate samples from a multivariate Student's t.
    https://en.wikipedia.org/wiki/Multivariate_t-distribution#Definition
    """

    # Dimensions of multivariate Student's t distribution.
    D = len(mean)

    rng = np.random.default_rng()
    x = rng.chisquare(degrees_of_freedom, n_samples) / degrees_of_freedom

    Z = rng.multivariate_normal(
        np.zeros(D),
        cov_matrix,
        n_samples,
    )

    X = mean + Z / np.sqrt(x)[:, np.newaxis]
    return X


class NystroemFeaturesSampler(BaseEstimator, TransformerMixin):
    """Sample Nystroem features."""

    def __init__(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = RBF(),
        n_random_features: int = 100,
    ) -> None:
        # _kernel -> kernel due to SKlearn compatibility
        self.kernel = kernel
        self.component_indices_ = None
        # J
        self._X_reduced = None
        # W
        self._reduced_kernel_matrix = None
        # (W+)^1/2
        self._sqrtm_pinv_reduced_kernel_matrix = None
        # Needed for compatibility
        self.n_random_features = n_random_features

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None  # Compatibility
    ) -> np.ndarray:
        """Precompute auxiliary quantities for Nystroem features."""
        n_instances = len(X)
        # Check dimensions to avoid selecting more features than
        # possible
        n_random_features = self.n_random_features
        if self.n_random_features > n_instances:
            n_random_features = n_instances

        # Sample subset of training instances.
        rng = np.random.default_rng()
        self.component_indices_ = rng.choice(
            range(n_instances),
            size=n_random_features,
            replace=False,
        )

        self._X_reduced = X[self.component_indices_, :]

        # Compute reduced kernel matrix.
        self._reduced_kernel_matrix = self.kernel(self._X_reduced, self._X_reduced)
        self._reduced_kernel_matrix = (
            self._reduced_kernel_matrix + self._reduced_kernel_matrix.T
        ) / 2.0  # enforce symmetry of kernel matrix

        # Compute auxiliary quantities.
        self._sqrtm_pinv_reduced_kernel_matrix = sp.linalg.sqrtm(
            np.linalg.pinv(self._reduced_kernel_matrix, rcond=1.0e-6, hermitian=True)
        )

        # Check that complex part is negligible and eliminate it
        if np.iscomplexobj(self._sqrtm_pinv_reduced_kernel_matrix):
            threshold_imaginary_part = 1.0e-6
            max_imaginary_part = np.max(
                np.abs(np.imag(self._sqrtm_pinv_reduced_kernel_matrix))
            )
            # if max_imaginary_part > threshold_imaginary_part:
            # warnings.warn("Maximum imaginary part is {}".format(max_imaginary_part))

            self._sqrtm_pinv_reduced_kernel_matrix = np.real(
                self._sqrtm_pinv_reduced_kernel_matrix
            )

    def approximate_kernel_matrix(
        self,
        X: np.ndarray,
        # Needed for sklearn compatibility
        X_prime: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Approximate the kernel matrix using Nystroem features."""

        X_features = self.fit_transform(X)
        return X_features @ X_features.T

    def fit_transform(
        self,
        X: np.ndarray,
        X_prime: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Nyström features."""
        self.fit(X)
        return self.transform(X)

    def transform(self, X_prime: np.ndarray) -> np.ndarray:
        """Compute Nystroem features with precomputed quantities."""
        J = self.kernel(X_prime, self._X_reduced)
        return J @ self._sqrtm_pinv_reduced_kernel_matrix


def demo_kernel_approximation_features(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_sampler_class: Union[
        Type[RandomFeaturesSampler], Type[NystroemFeaturesSampler]
    ],
    features_sampler_kwargs: np.ndarray,
    n_features: np.array,
) -> None:
    """Kernel approximation using random sampled features.
    Either RFF or Nyström features."""
    n_plots = len(n_features) + 1
    fig, axes = plt.subplots(1, n_plots)
    fig.set_size_inches(15, 4)
    font = {"fontname": "arial", "fontsize": 18}

    kernel_matrix = kernel(X, X)
    axes[0].imshow(kernel_matrix, cmap=plt.cm.Blues)
    axes[0].set_title("Exact kernel", **font)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for n_f, ax in zip(n_features, axes[1:]):
        print("# of features = ", n_f)

        features_sampler = features_sampler_class(
            **features_sampler_kwargs, n_random_features=n_f
        )

        X_features = features_sampler.fit_transform(X)
        kernel_matrix_approx = X_features @ X_features.T

        ax.imshow(kernel_matrix_approx, cmap=plt.cm.Blues)

        err_approx = kernel_matrix - kernel_matrix_approx
        err_mean = np.mean(np.abs(err_approx))
        err_max = np.max(np.abs(err_approx))

        ax.set_xlabel(
            "err (mean) = {:.4f} \n err (max) = {:.4f}".format(err_mean, err_max),
            **font
        )

        ax.set_title("{} features".format(n_f), **font)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    plt.show()


def plot_kernel_error(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_sampler_class: Union[
        Type[RandomFeaturesSampler], Type[NystroemFeaturesSampler]
    ],
    features_sampler_kwargs: np.ndarray,
    n_features: np.ndarray,
    kernel_name: str,
) -> None:
    # Initialize variables
    kernel_matrix = kernel(X, X)
    errors = np.array([], dtype=float)

    for n_f in n_features:
        # Create sampler
        features_sampler = features_sampler_class(
            **features_sampler_kwargs, n_random_features=n_f
        )

        # Compute approximation
        X_features = features_sampler.fit_transform(X)
        kernel_matrix_approx = X_features @ X_features.T

        # Compute error
        mean_error = np.mean(np.abs(kernel_matrix - kernel_matrix_approx))
        errors = np.append(errors, mean_error)

    # Plotting
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    font = {"fontname": "arial", "fontsize": 18}
    ax.plot(n_features, errors, color="deepskyblue", label=r"kernel error")
    ax.set_title("{} kernel error study".format(kernel_name), **font)

