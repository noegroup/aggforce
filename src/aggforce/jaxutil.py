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

