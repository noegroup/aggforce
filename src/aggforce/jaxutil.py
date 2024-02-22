"""Provides basic jax tools used in other submodules.

This module should not have dependencies on other package submodules.
"""
from functools import partial
from typing import Union, Callable
import jax
import jax.numpy as jnp


@jax.jit
def trjdot(points: jax.Array, factor: jax.Array) -> jax.Array:
    """Perform a JAX matrix product with mdtraj-style arrays and a matrix.

    NOTE: This function is similar to others in this package, but applies to JAX
    arrays

    Functionality is most easily described via an example:
        Molecular positions (and forces) are often represented as arrays of
        shape (n_steps,n_sites,n_dims). Other places in the code we often
        transform these arrays to a reduced (coarse-grained) resolution where
        the output is (n_steps,n_cg_sites,n_dims).

        (When linear) the relationship between the old (n_sites) and new
        (n_cg_sites) resolution can be described as a matrix of size
        (n_sites,n_cg_sites). This relationship is between sites, and is
        broadcast across the other dimensions. Here, the sites are contained in
        points, and the mapping relationship is in factor.

        However, we cannot directly use dot products to apply such a matrix map.
        This function applies this factor matrix as expected, in spirit of
        (points * factor).

        Additionally, if instead the matrix mapping changes at each frame of the
        trajectory, this can be specified by providing a factor of shape
        (n_steps,n_cg_sites,n_sites). This situation is determined by
        considering the dimension of factor.

    Arguments:
    ---------
    points (jnp.DeviceArray):
        3-dim array of shape (n_steps,n_sites,n_dims). To be mapped using
        factor.
    factor (jnp.DeviceArray):
        2-dim array of shape (n_cg_sites,n_sites) or 3-dim array of shape
        (n_steps,n_cg_sites,n_sites). Used to map points.

    Returns:
    -------
    jnp.DeviceArray of shape (n_steps,n_cg_sites,n_dims) contained points mapped
    with factor.
    """
    # knp einsum doesn't seem to accept the same path optimization directions
    # as np einsum, so we just pass "greedy"
    if len(factor.shape) == 2:
        return jnp.einsum("tfd,cf->tcd", points, factor, optimize="greedy")
    if len(factor.shape) == 3:
        return jnp.einsum("...fd,...cf->...cd", points, factor, optimize="greedy")
    raise ValueError("Factor matrix is an incompatible shape.")


def abatch(
    # mypy seems to not do partial signatures, but this function should take a first
    # argument of a jax.Array, other args satisfied via args/kwargs sharing
    func: Callable[..., jax.Array],
    arr: jax.Array,
    chunk_size: Union[None, int],
    *args,
    **kwargs,
) -> jax.Array:
    """Transparently apply a function over chunks of array.

    The results of func(arr) are computed by evaluating func(chunk), where chunk
    is a smaller piece of arr.

    NOTE: This function uses JAX calls.

    Arguments:
    ---------
    func (callable):
        Function applied to chunks of arr. Receives args/kwargs upon each
        invocation. Func (with args/kwargs) must be able to be applied to each
        chunk without changing the collective results (for example, it may be a
        vectorization of a per-frame function).
    arr (jnp.DeviceArray):
        Data to pass to func.
    chunk_size (positive integer):
        Size of array chunks (slicing across first index) to pass to func.
    *args:
        Passed to func at each invocation.
    **kwargs:
        Passed to func at each invocation.

    Returns:
    -------
    The results of func(arr) as computed by evaluating func(chunk).
    """
    if chunk_size is None or chunk_size >= arr.shape[0]:
        return func(arr, *args, **kwargs)
    n_chunks = jnp.ceil(len(arr) / chunk_size).astype(jnp.int32)
    arrs = jnp.array_split(arr, n_chunks)
    return jnp.vstack([func(subarr, *args, **kwargs) for subarr in arrs])


@partial(
    jax.jit,
    inline=True,
    static_argnames=["return_matrix", "return_displacements", "square"],
)
def distances(
    xyz: jax.Array,
    cross_xyz: Union[jax.Array, None] = None,
    return_matrix: bool = True,
    return_displacements: bool = False,
    square: bool = False,
) -> jax.Array:
    """Calculate differentiable distances for each frame in a trajectory.

    Returns an array where each slice is the distance matrix of a single frame
    of an argument.

    NOTE: This function is similar to others in this package, but applies to JAX
    arrays.

    Arguments:
    ---------
    xyz (jnp.DeviceArray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    cross_xyz (jnp.DeviceArray or None):
        An array describing the Cartesian coordinates of a different system over
        time or None; assumed to be of shape (n_steps,other_n_sites,n_dim). If
        present, then the returned distances are those between xyz and cross_xyz
        at each frame.  If present, return_matrix must be truthy.
    return_matrix (boolean):
        If true, then complete (symmetric) distance matrices are returned; if
        false, the upper half of each distance matrix is extracted, flattened,
        and then returned.
    return_displacements (boolean):
        If true, then instead of a distance array, an array of displacements is
        returned.
    square (boolean):
        If true, we return the square of the euclidean distance.

    Returns:
    -------
    Returns jnp.DeviceArray, where the number of dimensions and size depend on
    the arguments.

    If return_displacements is False:
        If return_matrix and cross_xyz is None, returns a 3-dim jnp.DeviceArrays
        of shape (n_steps,n_sites,n_sites), where the first index is the time
        step index and the second two are site indices. If return_matrix and
        cross_xyz is not None, then an array of shape
        (n_steps,other_n_sites,n_sites) is returned. If not return_matrix,
        return a 2-dim array (n_steps,n_distances), where n_distances indexes
        unique distances.
    else:
        return_matrix must be true, and a 4 dimensional array is returned,
        similar to the shapes above but with an additional terminal axis for
        dimension.
    """
    if cross_xyz is not None and not return_matrix:
        raise ValueError("Cross distances only supported when return_matrix is truthy.")
    if return_displacements and not return_matrix:
        raise ValueError("Displacements only supported when return_matrix is truthy.")

    if cross_xyz is None:
        displacement_matrix = xyz[:, None, :, :] - xyz[:, :, None, :]
    else:
        displacement_matrix = xyz[:, None, :, :] - cross_xyz[:, :, None, :]
    if return_displacements:
        return displacement_matrix
    if square:
        distance_matrix = (displacement_matrix**2).sum(axis=-1)
    else:
        distance_matrix = jnp.linalg.norm(displacement_matrix, axis=-1)
    if return_matrix:
        return distance_matrix
    n_sites = distance_matrix.shape[-1]
    indices0, indices1 = jnp.triu_indices(n_sites, k=1)
    subsetted_distances = distance_matrix[:, indices0, indices1]
    return subsetted_distances
