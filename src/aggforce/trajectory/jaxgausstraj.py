"""Jax-based Trajectory Augmenters."""
from typing import List, TypeVar, Optional, Union, Tuple, Callable, Final

from jax import Array, grad, vmap
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats.multivariate_normal import logpdf as jglogpdf
import numpy as np

from .augment import Augmenter

A = TypeVar("A")


def _ident(x: A, /) -> A:
    """Identity."""
    return x


class _perframe_wrap:
    """Transforms callable acting on chunks to one acting on frames.

    If I have a function that acts on trajectory arrays of size
    (n_frames,n_sites,n_dims), this class creates a callable that acts arrays of size
    (n_sites,n_dims), where the output the function evaluated on an array of shape
    (0,n_sites,n_dims) and then index along the first axis.

    This is needed for vmap calls in this module.
    """

    def __init__(self, call: Callable[[Array], Array]) -> None:
        """Initialize.

        call is the callable to be wrapped.
        """
        self.call = call

    def __call__(self, target: Array) -> Array:
        expanded_target = target[None, ...]
        return self.call(expanded_target)[0]


# we manipulate jax functions to create a function that provides the needed log
# derivatives.

# Ultimately, we need a function that provides the following log derivatives.
# Our conditional density is:
# g(y|x) := \propto \exp[ -(y-Ax)^T E^{-1} (y-Ax) ]
# where y is the generated position, x is the position of the real particles,
# A describes a linear mapping operation and E is a preset covariance matrix.
# We must obtain `grad log [g(y|x)]` where the gradient is with respect to both
# y and x.

# construct this function step by step.
# _mvgaussian_prefunc_logpdf
# gives the required log-density. The action of `A` is encapsulated in `pre_func`.


def _mvgaussian_prefunc_logpdf(
    variate: Array, pre_mean: Array, pre_func: Callable[[Array], Array], cov: Array
) -> Array:
    mean = pre_func(pre_mean)
    # mypy may be correct here for some corner cases of arguments?
    return jglogpdf(variate, mean, cov)  # type: ignore [return-value]


# _mvgaussian_prefunc_logpdf_grad differentiates this function w.r.t. the first
# two arguments: first y, then x.
# obtain partial gradient with respect to the first two arguments: variate and pre_mean.
_mvgaussian_prefunc_logpdf_grad = grad(_mvgaussian_prefunc_logpdf, argnums=(0, 1))

# _mggaussian_prefunc_logpdf_grad operates on a single variate for a single
# distribution. We vmap over the variate and mean to create a function that
# operates on a array of variantes and an array of distinct means, but with
# shared premap and cov matrix.
_mvgaussian_prefunc_logpdf_grad_vec = vmap(
    _mvgaussian_prefunc_logpdf_grad, in_axes=(0, 0, None, None), out_axes=0
)


class JCondNormal(Augmenter):
    r"""Augmenter that adds 0-mean Gaussian noise to mapped positions.

    Equivalently, this function creates the following conditional density:
    ```
    g(y|x) := \propto \exp[ -(y-Ax)^T E^{-1} (y-Ax) ]
    ```
    where `A` is a matrix specified by a Linear Map object and E is a given
    covariance matrix. E can be set via a scalar to be diagonal.

    premap is a callable which is applied to _flattened_ forms of the input
    coordinates. See _flatten and _unflatten for the flattening operation.
    It typically reduces the dimension of the input. postmap is applied to the
    returned forces/scores on the input coordinates, and again must act
    on flat arrays. While premap is usually specified, postmap only has
    use in cross resolution cases and should be used with extreme care: the
    gradients returned with postmap set to None are more intuitive.

    Note:
    ----
    This object uses Jax for derivatives, but all public methods/attributes
    use numpy hinting.

    Attributes/Methods:
    ----------
    sample:
        Provides augmenting samples.
    log_gradient:
        Gives the log gradients on both x and y.
    cov:
        Covariance matrix. If a scalar is given at initialization, cov is None
        until the first .sample call in order to learn the required
        dimension.

    """

    # Dimension of space each physical and augmenting particle resides in
    # used for reshaping arrays.
    n_dim: Final = 3

    def __init__(
        self,
        cov: Union[float, np.ndarray],
        premap: Optional[Callable[[Array], Array]] = None,
        source_postmap: Optional[Callable[[Array], Array]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        cov:
            Specifies the covariance matrix of the added gaussian noise. Note
            that this must be of shape (n_particles*n_dim,n_particles*n_dim).
        premap:
            Callable object used when creating the augmenting variables.
            Note that the dimension of the output of this function controls
            the dimension of the augmenting variables. See class description.
        source_postmap:
            Callable that is applied to the forces return on the source particles
            in log_gradient. This option
        seed:
            Seed for jax random number generation. If None, a random integer
            from numpy is used.

        """
        if premap is None:
            self.premap: Callable[[Array], Array] = _ident
        else:
            self.premap = premap
        if source_postmap is None:
            self.source_postmap: Callable[[Array], Array] = _ident
        else:
            self.source_postmap = source_postmap
        if seed is None:
            true_seed = np.random.default_rng().integers(low=0, high=int(1e6))
        else:
            true_seed = seed
        self._rkey, _ = jrandom.split(jrandom.PRNGKey(true_seed))
        self._cov = cov
        # if cov is a float, we need to defer creating the covariance matrix until
        # we see the dimensionality of samples.
        if isinstance(cov, Array):
            self.cov: Optional[Array] = cov
        else:
            self.cov = None

    def sample(self, source: np.ndarray) -> np.ndarray:
        """Generate Gaussian samples from an array of means.

        Arguments:
        ---------
        source:
            Slices along the first index of source give means, each of which is
            specifies a Gaussian to sample from. All variances of the samples Gaussians
            are given by self.cov.

        Returns:
        -------
        np.ndarray, where each slice along the leading axis is a Gaussian variate.

        Notes:
        -----
        This method expects and returns numpy arrays.

        """
        flattened = self._flatten(jnp.asarray(source))
        means = self.premap(flattened)
        return np.asarray(self._unflatten(self._sample(means)))

    def log_gradient(
        self, source: np.ndarray, generated: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate log gradients.

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

        Note:
        ----
        This method expects and returns numpy arrays.

        """
        flat_source = self._flatten(jnp.asarray(source))
        flat_generated = self._flatten(jnp.asarray(generated))

        if self.cov is None:
            raise ValueError(
                "Cannot generate log gradients without cov. Either specify"
                " cov at init, or call sample prior to log_gradient."
            )
        else:
            per_frame_premap = _perframe_wrap(self.premap)
            flat_lgrads = _mvgaussian_prefunc_logpdf_grad_vec(
                flat_generated, flat_source, per_frame_premap, self.cov
            )
            variate_lgrad = self._unflatten(flat_lgrads[0])
            source_lgrad = self._unflatten(flat_lgrads[1])

        post_source_lgrad = self.source_postmap(source_lgrad)

        return (np.asarray(post_source_lgrad), np.asarray(variate_lgrad))

    def _sample(self, means: Array, vectorized: bool = True) -> Array:
        """Generate Gaussian samples given array of means.

        Arguments:
        ---------
        means:
            Collection of Array instances. Each entry is a mean of a
            Gaussian used to generate the returned variates. Should be of shape
            (n_means,n_flat_dims).
        vectorized:
            Whether to use a reduced number of jax calls. False is only useful
            for debugging.

        Returns:
        -------
        2-Array where leading dimension indexes individual generated variates.

        Notes:
        -----
        If self.cov has not been set, this method sets it by looking at the
        dimension of the first means entry.

        """
        # if we have yet to create cov matrix, use first element of means to
        # determine its size.
        if self.cov is None:
            self.cov = jnp.diag(jnp.repeat(self._cov, repeats=len(means[0])))

        if vectorized:
            keys = jrandom.split(self._rkey, num=2)
            self._rkey = keys[0]
            data = jrandom.multivariate_normal(
                key=keys[1], mean=means, cov=self.cov[None, :]
            )
        else:
            keys = jrandom.split(self._rkey, num=len(means) + 1)
            self._rkey = keys[0]
            variates: List[Array] = []
            for mean, key in zip(means, keys[1:]):
                variates.append(
                    jrandom.multivariate_normal(key=key, mean=mean, cov=self.cov)
                )
            data = jnp.stack(variates, axis=0)
        return data

    def _flatten(self, array: Array) -> Array:
        """Flatten arrays of shape (n_f,n_p,n_dim) to (n_f,n_p*n_dim)."""
        old_shape = array.shape
        assert len(old_shape) == 3
        assert old_shape[-1] == self.n_dim
        return jnp.reshape(
            a=array, newshape=(old_shape[0], old_shape[1] * old_shape[2])
        )

    def _unflatten(self, array: Array) -> Array:
        """Undoes the action of _flatten."""
        old_shape = array.shape
        assert len(array.shape) == 2
        return jnp.reshape(
            a=array, newshape=(old_shape[0], old_shape[1] // self.n_dim, self.n_dim)
        )
