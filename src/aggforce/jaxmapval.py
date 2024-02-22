r"""Methods for validation and summarizing maps.

We focus on calculate the projection of forces along basis functions and the difference
in force residuals between pairs of force-fields.

This module requires JAX.

The function descriptions in this section make references to coarse-graining
(CG) terminology and ideas. For an introduction see the following publications:

Noid, William George, et al. "The multiscale coarse-graining method. I. A
rigorous bridge between atomistic and coarse-grained models." The Journal of
chemical physics 128.24 (2008): 244114.

Rudzinski, Joseph F., and W. G. Noid. "Coarse-graining entropy, forces, and
structures." The Journal of chemical physics 135.21 (2011): 214101.
"""

from typing import TypeVar, Union, Callable, Iterable, overload, Literal, List
import numpy as np
import numpy.random as r
import jax
from .agg import force_smoothness
from .jaxutil import distances
from .qp.jaxfeat import clipped_gauss

ArrayT = TypeVar("ArrayT", bound=Union[jax.Array, np.ndarray])


def random_uniform_forces(
    positions: np.ndarray,
    scale: float = 1.0,
    randg: Union[r.Generator, None] = None,
) -> np.ndarray:
    r"""Obtain forces of a random linear force-field applied to a trajectory.

    This function evaluates a linear force-field which assigns a single unique
    3-vector to each site at each frame. That is, all sites in all frames have
    the same force (same direction and magnitude). This force has magnitude
    scale and points in a shared random direction, which is set using randg.

    Arguments:
    ---------
    positions (array, jnp.DeviceArray or np.ndarray):
        3-d array containing positions as a function of time. Assumed to be of
        shape (n_steps,n_sites,n_dims).
    scale (positive float):
        the magnitude of the random force applied to each site.
    randg (None or r.Generator instance):
        random number generator used to define the force direction. To make the
        direction deterministic, supply a Generator instance with a fixed seed.

    Returns:
    -------
    3-dimensional nd.ndaray containing the forces for each frame in positions.
    Has shape (n_frames,n_sites,n_dims).
    """
    if randg is None:
        randg = r.default_rng()
    shape = positions.shape
    x, y, z = 2 * randg.random(size=3) - 1
    force = np.array([x, y, z])
    force /= ((force**2).sum()) ** (0.5)
    force *= scale
    expanded_force = force[None, None, :]
    parti_tiled_force = np.repeat(
        expanded_force,
        repeats=shape[0],
        axis=0,
    )
    tiled_force = np.repeat(
        parti_tiled_force,
        repeats=shape[1],
        axis=1,
    )
    return tiled_force


def rsqpg_forces(
    positions: Union[jax.Array, np.ndarray],
    inner: float,
    outer: float,
    width: float,
    randg: Union[r.Generator, None] = None,
    sq_args: bool = True,
) -> jax.Array:
    r"""Calculate the forces of a random force-field.

    Estimates the forces of a random Gaussian force-field for each frame in
    a trajectory.

    randg is used to generate a random number in between inner and outer; this
    number is used as the offset for a Gaussian. This single Gaussian is applied
    to every pairwise distance of every frame, and the Gaussian values for each
    frame are summed to give that frame's energy. This energy is differentiated
    with respect to positions (and multiplied by -1) to provide forces for each
    frame.

    Arguments:
    ---------
    positions (array, jnp.DeviceArray or np.ndarray):
        3-d array containing positions as a function of time. Assumed to be of
        shape (n_steps,n_sites,n_dims).
    randg (None or r.Generator instance):
        random number generator used to define the Gaussian offset. To make the
        selection of width deterministic, supply a Generator instance with a
        fixed seed.
    inner (positive float):
        inner limit of possible offsets generated for the Gaussian. See sq_args.
    outer (positive float):
        outer limit of possible offsets generated for the Gaussian. See sq_args.
    width (positive float):
        width of the Gaussian
    sq_args (boolean):
        If truthy, inner, outer, and width are squared before they are used.

    Returns:
    -------
    3-dimensional jnp.DeviceArray containing the forces for each frame in
    positions. Has shape (n_frames,n_sites,n_dims).
    """
    if sq_args:
        outer = outer**2
        inner = inner**2
        width = width**2
    if randg is None:
        randg = r.default_rng()
    interval_width = outer - inner
    offset = randg.random() * interval_width + inner
    return sq_gaussian_forces(positions, offset, width)


@overload
def random_residual_shift(
    coords: np.ndarray,
    forces: np.ndarray,
    n_samples: int,
    randg: Union[r.Generator, None],
    method: Callable[..., jax.Array],  # see docstring
    average: Literal[True],
    **kwargs
) -> float:
    ...


@overload
def random_residual_shift(
    coords: np.ndarray,
    forces: np.ndarray,
    n_samples: int,
    randg: Union[r.Generator, None],
    method: Callable[..., jax.Array],  # see docstring
    average: Literal[False],
    **kwargs
) -> List[float]:
    ...


def random_residual_shift(
    coords: np.ndarray,
    forces: np.ndarray,
    n_samples: int = 1000,
    randg: Union[r.Generator, None] = None,
    method: Callable[..., jax.Array] = rsqpg_forces,  # see docstring
    average: bool = False,
    **kwargs
) -> Union[float, List[float]]:
    r"""Estimate the force roughness of generated force-fields.

    Calculates the force residual (i.e. force_smoothness) difference between a
    series of randomly generated force-fields and a flat force-field.

    This procedure is based on the following property:

    The configurational integral form of the force residual (here denoted R and
    evaluated via force_smoothness) evaluated for force-field G has a noise term
    and a noiseless term as so:
        R[G] = Noise + R_noiseless[G]
    The noise term does not depend on G. As a result, if you take the difference
    between R for two force-fields they cancel out:
        R[G_0] - R[G_1] = R_noiseless[G_0] - R_noiseless[G_1]
    Critically, the choice of force map only affects the Noise term. In other
    words, if we take the difference in two force residuals, the noise does not
    matter.

    This function calculates a series of such differences, which we refer to as
    shifts. First, n_sample different candidate force-fields are randomly
    generated using randg and method. Each of these candidate potentials are
    characterized via their force residual (through force_smoothness). Finally,
    each of these forces residuals have a reference force residual subtracted
    from them. The reference is the force residual of a constant-everywhere
    force-field (which as a force of zero everywhere).

    Arguments:
    ---------
    coords (np.ndarray):
        3-array that has the positions of the system as a function of time.
        Must be of shape (n_steps,n_sites,n_dims). This is fed as input into
        method. Should be the CG coordinates of the system.
    forces (np.ndarray):
        3-array that has the forces of the system as a function of time.  Must
        be of shape (n_steps,n_sites,n_dims). Should contain the mapped
        fluctuating CG forces (the output of a force map).
    n_samples (positive integer):
        number of basis force-fields to calculate the shift for. If average is
        falsey, then this is the length of the output.
    randg (None or r.Generator instance):
        random number generator used to define the basis functions. To make the
        selection of basis functions deterministic supply a Generator instance
        with a fixed seed.
    method (callable):
        callable used to generate the random force-fields. coords, randg and
        **kwargs are passed at each call, and it should return a callable that
        outputs forces for each frame if given coords.
    average (boolean):
        whether to average the shifts or return a list of all of the
        shifts.
    **kwargs:
        passed to method at each iteration.

    Returns:
    -------
    if average is truthy, returns a scalar (often as a JAX 0-array depending out the
    particular method used); else, a list of such scalars with an entry for each
    randomly generated force-field.
    """
    if randg is None:
        randg = r.default_rng()
    vals = []
    for _ in range(n_samples):
        trial_forces = method(coords, randg=randg, **kwargs)
        vals.append(force_smoothness(forces - trial_forces))
    if average:
        return sum(vals) / n_samples - force_smoothness(forces)
    else:
        fs = force_smoothness(forces)
        return [x - fs for x in vals]


@overload
def random_force_proj(
    coords: np.ndarray,
    forces: np.ndarray,
    n_samples: int,
    randg: Union[r.Generator, None],
    method: Callable[..., jax.Array],
    average: Literal[True],
    **kwargs
) -> float:
    ...


@overload
def random_force_proj(
    coords: np.ndarray,
    forces: np.ndarray,
    n_samples: int,
    randg: Union[r.Generator, None],
    method: Callable[..., jax.Array],
    average: Literal[False],
    **kwargs
) -> Iterable[float]:
    ...


def random_force_proj(
    coords: np.ndarray,
    forces: np.ndarray,
    n_samples: int = 1000,
    randg: Union[r.Generator, None] = None,
    method: Callable[..., jax.Array] = rsqpg_forces,
    average: bool = True,
    **kwargs
) -> Union[float, Iterable[float]]:
    r"""Perform MSCG projections on random basis functions.

    Generates n_samples different basis functions and projects forces on them.
    See mscg_ip for more details.

    Arguments:
    ---------
    coords (np.ndarray):
        3-array that has the positions of the system as a function of time.
        Must be of shape (n_steps,n_sites,n_dims).
    forces (np.ndarray):
        3-array that has the forces of the system as a function of time.  Must
        be of shape (n_steps,n_sites,n_dims).
    n_samples (positive integer):
        number of basis functions to project onto. If average if falsey, then
        this is the length of the output.
    randg (None or r.Generator instance):
        random number generator used to define the basis functions. To make the
        selection of basis functions deterministic, supply a Generator instance
        with a fixed seed.
    method (callable):
        Callable used to generate the random basis functions. coords, randg and
        **kwargs are passed at each call, and it should return a callable that
        satisfies the properties needed by mscg_ip.
    average (boolean):
        Whether to average the projections or return a list of all of the
        projections.
    kwargs:
        passed to method at each iteration via **.

    Returns:
    -------
    if average is truthy, returns a scalar (often as a JAX 0-array depending out the
    particular method used); else, a list of such scalars.
    """
    if randg is None:
        randg = r.default_rng()
    vals = []
    for _ in range(n_samples):
        trial_func = method(coords, randg=randg, **kwargs)
        vals.append(mscg_ip(forces, trial_func))
    if average:
        return sum(vals) / n_samples
    else:
        return vals


def mscg_ip(forces: ArrayT, funcs: ArrayT) -> float:
    r"""Perform a MSCG-like inner product.

    This function does an element-wise product of forces and funcs, sums over all
    dims, and then divides by the size of the first (0th) dimension.

    NOTE: funcs is an array of outputted function values, _not_ a callable.

    The action of this function is based on the following relationship:
        \int F(x) \dot G \circ M (x) = H
    where F(x) is the force map being applied to configuration x, G is a
    function of CG (mapped) coordinates, M is the configurational CG map, and
    the integral is performed over the fine-grained (FG) configurational
    ensemble. G is a function that has the same shaped output as the force map,
    and \dot multiplies all values of F and G element-wise and sums them at each
    configuration. \circ denotes function composition.

    Note that in order to correspond to the integral described above, funcs must
    be the output of a function of the CG (and not FG) coordinates. This
    function is still defined for general arrays, but does not correspond to the
    same type of sample-based approximation to the content described above.

    Arguments:
    ---------
    forces (3-array):
        3-array (jax or numpy) containing the forces associated each timestep of
        a trajectory. Must be of shape (n_steps,n_sites,n_dims).
    funcs (3-array):
        3-array (jax or numpy) containing the output of function G. Must be of
        shape (n_steps,n_sites,n_dims).  NOTE: This is _not_ a callable. It is
        an array representing the output of a suitable function.

    Returns:
    -------
    Returns a 0-dim array containing the value of the inner product. The exact
    type depends on the input types.
    """
    n_steps = forces.shape[0]
    return float((funcs * forces).sum() / n_steps)


# note that JAX peculiarities make a gaussian energy function that is a function
# of distances and not square distances return forces that are nans.
@jax.jit
def sq_gaussian_energies(
    positions: jax.Array, offset: float, width: float
) -> jax.Array:
    r"""Calculate per-frame energies of positions with Gaussian potential.

    A single Gaussian (with properties set by offset and width) is applied to
    each pairwise distance in each frame. This is then summed over each frame to
    produce the trajectory of energies.

    Arguments:
    ---------
    positions (jnp.DeviceArray):
        3-array with shape (n_frames, n_sites, n_dims) that contains the
        positions of the sites as a function of time.
    offset (float):
        sets the Gaussian offset through clipped_gauss
    width (float):
        sets the Gaussian width through clipped_gauss

    Returns:
    -------
    1-dimensional jnp.DeviceArray containing the energy of each frame.
    """
    distance_arr = distances(positions, return_matrix=True, square=True)
    return clipped_gauss(distance_arr, center=offset, width=width, clip=None).sum(
        axis=[1, 2]
    )


# forces corresponding to sq_gaussian_energies
sq_gaussian_forces = jax.jacrev(
    lambda positions, offset, width: -sq_gaussian_energies(
        positions=positions, offset=offset, width=width
    ).sum(),
    0,
)
