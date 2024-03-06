"""Extends LinearMaps for Jax operations."""
from typing import overload, TypeVar
from jax import Array
import jax.numpy as jnp
from numpy.typing import NDArray
import numpy as np
from .core import LinearMap
from ..jaxutil import trjdot as jtrjdot

ArrT = TypeVar("ArrT", NDArray, Array)


class JLinearMap(LinearMap):
    """Extends LinearMaps to map Jax arrays."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize.

        All argments are passed via super().
        """
        super().__init__(*args, **kwargs)

    @property
    def jax_standard_matrix(self) -> Array:
        """Return standard_matrix as a Jax array."""
        return jnp.asarray(self.standard_matrix)

    @overload
    def __call__(self, points: NDArray) -> NDArray:
        ...

    @overload
    def __call__(self, points: Array) -> Array:
        ...

    def __call__(self, points: ArrT) -> ArrT:
        r"""Apply map to a particular form of 3-dim array.

        Arguments:
        ---------
        points:
            3 dimensional of shape (n_steps,n_sites,n_dims). May be either a
            jax.Array or np.ndarray, and type is preserved in the returned
            value.

        Returns:
        -------
        Combines points along the n_sites dimension according to the internal
        map.

        Notes:
        -----
        This implementation is effectively identical to that in the parent class,
        but uses Jax operations.

        """
        if isinstance(points, np.ndarray):
            jpoints = jnp.asarray(points)
        else:
            jpoints = points
        transformed = jtrjdot(jpoints, self.jax_standard_matrix)
        if isinstance(points, np.ndarray):
            return np.asarray(transformed)
        else:
            return transformed

    @overload
    def flat_call(self, flattened: NDArray) -> NDArray:
        ...

    @overload
    def flat_call(self, flattened: Array) -> Array:
        ...

    def flat_call(self, flattened: ArrT) -> ArrT:
        """Apply map to pre-flattened array.

        Array is reshaped, mapped, and then reshaped.

        Arguments:
        ---------
        flattened:
            2-D array of shape (n_frames,n_fg_sites*n_dim). Likely created by
            flattening a matrix of shape (n_frames,n_fg_sites*n_dim).

        Returns:
        -------
        Returns mapped array of shape (n_frames,n_cg_sites*n_dim).

        Notes:
        -----
        This implementation is effectively identical to that in the parent class,
        but has an extended type signature due the extended call method.

        """
        shape = flattened.shape
        if len(shape) == 3:
            raise ValueError(f"Expected array of rank 2; got array with shape {shape}.")
        if flattened.shape[1] % self.n_dim != 0:
            raise ValueError(
                f"Array of shape {shape} can't be reshaped with dim of f{self.n_dim}."
            )
        reshaped = flattened.reshape(
            (flattened.shape[0], flattened.shape[1] // self.n_dim, self.n_dim),
        )
        transformed = self(reshaped)
        return transformed.reshape(
            (transformed.shape[0], transformed.shape[1] * transformed.shape[2]),
        )

    @property
    def T(self) -> "JLinearMap":
        """LinearMap defined by transpose of its standard matrix."""
        return JLinearMap(mapping=self.standard_matrix.T)

    def __matmul__(self, lm: "LinearMap", /) -> "JLinearMap":
        """LinearMap defined by multiplying the standard_matrix's of arguments."""
        return JLinearMap(mapping=self.standard_matrix @ lm.standard_matrix)

    def __rmul__(self, c: float, /) -> "JLinearMap":
        """LinearMap defined by multiplying the standard_matrix's with a coefficient."""
        return JLinearMap(mapping=c * self.standard_matrix)

    def __add__(self, lm: "LinearMap", /) -> "JLinearMap":
        """LinearMap defined by adding standard_matrices."""
        return JLinearMap(mapping=self.standard_matrix + lm.standard_matrix)

    @classmethod
    def from_linearmap(cls, lm: LinearMap, /) -> "JLinearMap":
        """Create JLinearMap from LinearMap."""
        return JLinearMap(mapping=lm.standard_matrix)
