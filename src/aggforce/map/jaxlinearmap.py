"""Extends LinearMaps for Jax operations."""
from typing import overload, TypeVar, Tuple, Union
from functools import partial
from jax import Array, jit
import jax.numpy as jnp
from numpy.typing import NDArray
import numpy as np
from .core import LinearMap
from ..jaxutil import trjdot as jtrjdot

ArrT = TypeVar("ArrT", NDArray, Array)


@partial(jit, static_argnames="nan_handling")
def _trjdot_worker(
    factor: Array, points: Array, nan_handling: bool
) -> Tuple[Array, Array]:
    """Help apply internal trjdot transforms.

    If nan_handling is false, applies trjdot and returns
    a tuple with both entries the same result. If true, the first
    entry in the tuple is the result of setting nans to 0, and the second
    result is setting nans to 1.
    """
    if nan_handling:
        input_matrix_0 = jnp.nan_to_num(
            points,
            nan=0.0,
        )
        input_matrix_1 = jnp.nan_to_num(
            points,
            nan=1.0,
        )
        result_0 = jtrjdot(input_matrix_0, factor)
        result_1 = jtrjdot(input_matrix_1, factor)
        return (result_0, result_1)
    else:
        result = jtrjdot(points, factor)
        return (result, result)


class JLinearMap(LinearMap):
    """Extends LinearMaps to map Jax arrays."""

    def __init__(self, *args, bypass_nan_check: bool = False, **kwargs) -> None:
        """Initialize.

        Arguments:
        ---------
        *args:
            Passed via super().
        bypass_nan_check:
            If true, we check to see if infs were generated when mapping matrices
            with a nan check (similar to LinearMap behavior). If not, we do not;
            this often must be set to false to be wrapped in a jit call.
        **kwargs:
            Passed via super().

        """
        super().__init__(*args, **kwargs)
        self.bypass_nan_check = bypass_nan_check
        self._jax_standard_matrix = jnp.asarray(self.standard_matrix)

    @property
    def jax_standard_matrix(self) -> Array:
        """Return standard_matrix as a Jax array."""
        return self._jax_standard_matrix

    @overload
    def __call__(self, points: NDArray) -> NDArray:
        ...

    @overload
    def __call__(self, points: Array) -> Array:
        ...

    def __call__(self, points: Union[NDArray, Array]) -> Union[NDArray, Array]:
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
        but will behave differently if an invalid map is applied to

        """
        if isinstance(points, np.ndarray):
            numpy_input = True
            jpoints = jnp.asarray(points)
        else:
            numpy_input = False
            jpoints = points

        result, sec_result = _trjdot_worker(
            factor=self.jax_standard_matrix,
            points=jpoints,
            nan_handling=self.handle_nans,
        )
        if (not self.bypass_nan_check) and self.handle_nans:
            if not jnp.allclose(result, sec_result, atol=self.nan_check_threshold):
                raise ValueError(
                    "NaN handling is on and multiplication tried to use "
                    "a NaN value. Check the input array and "
                    "standard_matrix."
                )
        if numpy_input:
            return np.asarray(result)
        else:
            return result

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
        return JLinearMap(
            mapping=self.standard_matrix.T,
            bypass_nan_check=self.bypass_nan_check,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )

    def __matmul__(self, lm: "LinearMap", /) -> "JLinearMap":
        """LinearMap defined by multiplying the standard_matrix's of arguments."""
        return JLinearMap(
            mapping=self.standard_matrix @ lm.standard_matrix,
            bypass_nan_check=self.bypass_nan_check,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )

    def __rmul__(self, c: float, /) -> "JLinearMap":
        """LinearMap defined by multiplying the standard_matrix's with a coefficient."""
        return JLinearMap(
            mapping=c * self.standard_matrix,
            bypass_nan_check=self.bypass_nan_check,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )

    def __add__(self, lm: "LinearMap", /) -> "JLinearMap":
        """LinearMap defined by adding standard_matrices."""
        return JLinearMap(
            mapping=self.standard_matrix + lm.standard_matrix,
            bypass_nan_check=self.bypass_nan_check,
            handle_nans=self.handle_nans,
            nan_check_threshold=self.nan_check_threshold,
        )

    @classmethod
    def from_linearmap(
        cls, lm: LinearMap, /, bypass_nan_check: bool = False
    ) -> "JLinearMap":
        """Create JLinearMap from LinearMap."""
        return JLinearMap(
            mapping=lm.standard_matrix,
            bypass_nan_check=bypass_nan_check,
            handle_nans=lm.handle_nans,
        )

    def to_linearmap(self) -> LinearMap:
        """Create normal LinearMap from the current object."""
        return LinearMap(mapping=self.standard_matrix, handle_nans=self.handle_nans)
