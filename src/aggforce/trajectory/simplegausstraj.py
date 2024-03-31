"""Numpy-based Trajectory augmenter that is faster for simple cases."""

from typing import Optional, Union, Tuple, Final

import numpy as np
from numpy.typing import DTypeLike

from .augment import Augmenter

_UNSET: Final = object()


class SimpleCondNormal(Augmenter):
    r"""Augmenter that adds 0-mean Gaussian noise to mapped positions.

    The added noise is multivariate, but must have a diagonal covariance matrix:
    we add isotropic and independent Gaussian noise variates. The variances is
    specified via a scalar.

    Attributes/Methods:
    ------------------
    sample:
        Provides augmenting samples.
    log_gradient:
        Gives the log gradients on both x and y.
    var:
        Variance scalar.

    """

    def __init__(
        self,
        var: float,
        seed: Optional[int] = None,
        dtype: Union[DTypeLike, object] = _UNSET,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        var:
            Variance of added Gaussian noise.
        seed:
            Seed passed to np.default_rng init.
        dtype:
            numpy dtype-compatible object that specifies the precision of method output.
        """
        self.var = var
        self._rng = np.random.default_rng(seed)
        if dtype is _UNSET:
            self.dtype: np.dtype = np.dtype(np.float32)
        else:
            self.dtype = np.dtype(dtype)  # type: ignore [arg-type]

    def sample(self, source: np.ndarray) -> np.ndarray:
        """Generate Gaussian samples from a tensor of means.

        Arguments:
        ---------
        source (ndarray, shape [*, n_particles, n_dim]):
            Slices along the first index of source give means, each of which is
            specifies a Gaussian to sample from.

        Returns:
        -------
        ndarray, where each slice along the leading axis is a Gaussian variate.

        """
        # would premap here # means = self.premap(source)
        means = source
        stdvar = np.sqrt(self.var)
        noise = stdvar * self._rng.standard_normal(source.shape, dtype=self.dtype)
        return (means + noise).astype(self.dtype, copy=False)

    def log_gradient(
        self, source: np.ndarray, generated: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate gradients of the log pdf.

        Arguments:
        ---------
        source:
            Array of the positions of the real particles. Should be of shape
            (n_frames,n_sites,n_dims).
        generated:
            Array of the positions of the real particles. Should be of shape
            (n_frames,n_generated_sites,n_dims). n_dims and n_frames
            should match those of source. n_generated_sites must match the dimension
            implied by premap.

        Returns:
        -------
        Tuple of arrays: the first element is the log gradients of the conditional
        density with respect to the real particle positions, and the second is the
        log gradients with respect to the generated particle positions.

        """
        # since premap = id
        mean = source
        # we want d(logpdf) / d(source) and d(jglogpdf) / d(variate)
        # note that log_norm = -0.5y^T @ \Sigma^-1 @ y
        #                      - n/2*\log(2\pi) - 0.5\log|\Sigma|
        # we only need the derivative coming from the quadratic form
        # which is d(log_norm)/dy = -(\Sigma^-1 + \Sigma^-1^T)/2 @ y = -\Sigma^-1 y
        # since y = variate - source
        # then d(logpdf) / d(source) = \Sigma^-1 y
        #       and d(logpdf) / d(variate) = -\Sigma^-1 y
        inv_var = 1.0 / self.var
        derivative = (-inv_var * (generated - mean)).astype(self.dtype, copy=False)
        return -derivative, derivative

    def astype(
        self, dtype: DTypeLike, *args, **kwargs  # noqa: ARG002
    ) -> "SimpleCondNormal":
        """Return instance with a specified dtype.

        See dtype argument of init for more information. Note that args and kwargs are
        ignored; they are provided for compatibility with a numpy.dtype call.

        Arguments:
        ---------
        dtype:
            Passed to init of new instance.
        *args:
            Ignored
        **kwargs:
            Ignored

        Returns:
        -------
        A SimpleCondNormal instance with the dtype set.
        """
        new_instance = self.__class__(
            var=self.var,
            dtype=dtype,
        )
        return new_instance
