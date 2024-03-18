"""Numpy-based Trajectory augmenter that is faster for simple cases"""

from typing import List, TypeVar, Optional, Union, Tuple, Callable, Final

import numpy as np
from numpy.typing import DTypeLike

from .augment import Augmenter
from ..map import LinearMap
from .jaxgausstraj import _ident, JCondNormal

_UNSET: Final = object()


def _is_identity_map(fn):
    if fn == _ident:
        return True
    elif isinstance(fn, LinearMap):
        # check whether the matrix is close to id
        dim = fn.standard_matrix.shape[0]
        id_ = np.eye(dim)
        if np.allclose(fn.standard_matrix, id_):
            return True
    return False


class SimpleCondNormal(Augmenter):
    r"""Augmenter that adds 0-mean Gaussian noise to mapped positions. This variant
    implements the same `Augmenter` interface as `JCondNormal` but aims to be faster
    in noising dataset operations.

    Essentially, we expect the covariance matrix to be `cov * I`, where `cov` has to be
    scalar. In other words, the noising process is adding isotropic and indenpendent
    Gaussian noises.

    One can call class method `from_jcondnormal` to convert from a `JCondNormal`
    instance, given that the arguments `cov` is scalar, the transform `premap` and
    `source_postmap` are both identity or a `LinearMap` that is close to identity.

    Attributes/Methods:
    ----------
    sample:
        Provides augmenting samples.
    log_gradient:
        Gives the log gradients on both x and y.
    cov:
        Variance scalar.

    """

    def __init__(
        self,
        cov: float,
        seed: Optional[int] = None,
        dtype: Union[DTypeLike, object] = _UNSET,
    ):
        self._cov = cov
        if np.array(self._cov).ndim != 0:
            raise NotImplementedError(
                f"`SimpleCondNormal` is only implemented for "
                f"scalar covariance instead of array shape "
                f"{self._cov.shape}."
            )
        self._inv_cov = 1 / cov
        self._rng = np.random.default_rng(seed)
        if dtype is _UNSET:
            self.dtype = np.float32
        else:
            self.dtype = np.dtype(dtype)  # type: ignore [arg-type]

    def sample(self, source: np.ndarray) -> np.ndarray:
        """Generate Gaussian samples from a tensor of means.

        Arguments:
        ---------
        source (tensor, shape [*, n_particles, n_dim]):
            Slices along the first index of source give means, each of which is
            specifies a Gaussian to sample from.

        Returns:
        -------
        torch.FloatTensor, where each slice along the leading axis is a Gaussian variate.

        """
        # means = self.premap(source) # ignored for current impl
        means = source
        noise = self._cov * self._rng.standard_normal(*source.shape, dtype=self.dtype)
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
        # note that log_norm = -0.5y^T @ \Sigma^-1 @ y - n/2*\log(2\pi) - 0.5\log|\Sigma|
        # we only need the derivative coming from the quadratic form
        # which is d(log_norm)/dy = -(\Sigma^-1 + \Sigma^-1^T)/2 @ y = -\Sigma^-1 y
        # since y = variate - source
        # then d(logpdf) / d(source) = \Sigma^-1 y and d(logpdf) / d(variate) = -\Sigma^-1 y
        derivative = (-self._inv_cov * (generated - mean)).astype(
            self.dtype, copy=False
        )
        return -derivative, derivative

    @classmethod
    def from_jcondnormal(cls, jcn: JCondNormal, /) -> "SimpleCondNormal":
        """Create TorchCondNormal from JCondNormal.
        constraints:
        - cov: only scalar float
        - premap: only accepting `aggforce.trajectory.jaxgausstraj._ident`
        - source_postmap: only accepting `aggforce.trajectory.jaxgausstraj._ident` or a
            `JLinear` with an almost identity standard_matrix
        - seed: ignored, since not possible to recover from an instance
        """
        # check the arguments
        if not _is_identity_map(jcn.premap):
            raise NotImplementedError(
                f"`SimpleCondNormal` is only implemented for "
                f"cases where `premap` is simply the identity "
                f"transform."
            )
        if not _is_identity_map(jcn.source_postmap):
            raise NotImplementedError(
                f"`SimpleCondNormal` is only implemented for "
                f"cases where `source_postmap` is simply the "
                f"identity transform."
            )
        return SimpleCondNormal(cov=jcn._cov, dtype=jcn.dtype)

    def astype(
        self, dtype: DTypeLike, *args, **kwargs  # noqa: ARG002
    ) -> "JCondNormal":
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
            cov=self._cov,
            dtype=dtype,
        )
        return new_instance
