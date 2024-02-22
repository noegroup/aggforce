"""Provides Jax routines to manipulate Map objects."""
from typing import Callable
from jax import Array
from jax.numpy import array
from ..jaxutil import trjdot
from .core import LinearMap


def jaxify_linearmap(
    lm: LinearMap,
    flattened: bool = True,
    n_dim: float = 3,
) -> Callable[[Array], Array]:
    """Turn a LinearMap object into a Jax-compatible callable.

    If flattened is True, then the derived function works on flattened arrays.
    For example, for a trajectory of shape (5,2,3) (5 frames, 2 particles, and a
    dimension of 3), the callable would expect input in the shape of (5,6)--- this
    is then reshaped internally, transformed, and then again reshaped before being
    returned. For example, if the internal map matrix if shape (1,2), then
    the output would be of shape (5,3).

    If flattened if False, then the input would be expected to be of shape (5,2,3)
    and the output would be (5,1,3).


    Arguments:
    ---------
    lm:
        LinearMap instance from which the standard_matrix is extracted and applied
        via trjdot.
    flattened:
        If true, input to the derived function is assumed to be a flattened
        with respect along the second and third indices and is reshaped before
        application. The output is then also flattened.
    n_dim:
        The size of the dimension in which each particle resides (this is almost
        always 3 in molecular dynamics). Only used if flattened is True.

    Returns:
    -------
    Callable that acts on Jax arrays and takes an optional perframe argument.
        perframe:
            If true, the input to the derived function is assumed to be missing its
            leading index (i.e., it is given single frames). It is expanded, mapped,
            and then the leading index is removed when the result is returned.

            This is mainly used for vmap calls.
    """
    matrix = array(lm.standard_matrix)

    def wrapped(mat: Array, perframe: bool = False) -> Array:
        if perframe:
            mat = mat[None, ...]
        if flattened:
            mat = mat.reshape((mat.shape[0], mat.shape[1] // n_dim, n_dim))
        result = trjdot(points=mat, factor=matrix)
        if flattened:
            result = result.reshape(
                (result.shape[0], result.shape[1] * result.shape[2])
            )
        if perframe:
            result = result[0]
        return result

    return wrapped
