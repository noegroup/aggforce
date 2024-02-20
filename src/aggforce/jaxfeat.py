"""JAX-based library for creating features for nonlinear map optimization."""

from typing import Union, Tuple, Iterable, Callable, Final
from functools import partial
import jax.numpy as jnp
import jax
import numpy as np
from .map import smear_map
from .linearmap import reduce_constraint_sets
from .featlinearmap import id_feat
from .map import LinearMap
from .constfinder import Constraints
from .featlinearmap import Features, KNAME_FEATS, KNAME_DIVS, KNAME_NAMES


DIVMETHOD_REORDER: Final = "reorder"
DIVMETHOD_BASIC: Final = "basic"


def gb_feat(
    points: np.ndarray,
    cmap: LinearMap,
    constraints: Constraints,
    outer: float,
    inner: float = 0,
    n_basis: int = 10,
    width: float = 1.0,
    dist_power: float = 0.5,
    batch_size: Union[None, int] = None,
    lazy: bool = True,
    div_method: str = DIVMETHOD_REORDER,
) -> Features:
    """Featurize each site via its distance to a mapped site.

    Each fine-grained site is characterized via its distance to the coarse-grained
    site at each frame.

    This function uses JAX for acceleration.

    At each frame, the distances between a coarse-grained site and each
    fine-grained site are calculated. These distances are "binned" by applying a
    series of Gaussians at multiple points along each distance. These binned
    distances are then associated separately to each atom (this is done using a
    one-hot encoding in the feature matrix).

    Fine-grained sites which are constrained together are assigned the same
    position before distance calculation and use the same one-hot slot; as a
    result, they have identical features at each frame.

    Arguments:
    ---------
    points (np.ndarray):
        Positions of the fine_grained trajectory. Assumed to have shape
        (n_frames,n_fg_sites,n_dims).
    cmap (map.LinearMap):
        Configurational map that links the fine-grained and coarse-grained
        resolutions.
    constraints (set of frozensets):
        Set of frozensets, each of which contains a set of fine-grained
        sites which have a molecular constraint applied. Constrained
        groups may overlap.
    outer (positive float):
        The largest distance to consider when making the grid of Gaussians.
    inner (non-negative float):
        The smallest distance to consider when making the grid of Gaussians.
    n_basis (positive integer):
        Number of Gaussian bins to use. Higher numbers are more expressive, but
        increase memory usage.
    width (positive integer):
        Controls the width of each Gaussian. Gaussians are roughly calculated as
        exp(-d**2/width), where d is the distance.
    dist_power (positive float):
        Controls the spacing and scaling of the Gaussians. Values greater than 1
        concentrate Gaussians towards outer, values between 0 and 1 concentrate
        Gaussians towards inner. Areas with more concentrated Gaussians also
        have Gaussians of less variance. See gaussian_dist_basis for more
        information.
    batch_size (positive integer):
        Number of trajectory frames to feed into JAX at once. Larger values are
        faster but use more memory.
    lazy (boolean):
        If truthy, generators of features and divs are returned; else, lists are
        returned.
    div_method (string):
        Determines how the divergence will be calculated; passed to
        gb_subfeat_jac as method.

    Returns:
    -------
    A dictionary with two three elements pairs:
        'feats': list/generator of feature mats (n_frames, n_fg_sites, total_n_feats)
            The features do not changed between frames.
        'divs': list/generator of div mats (n_frames, total_n_feats, n_dim).
            These are filled with zeros as the features do not change as a
            function of position.
        'names': None
    Each element of these lists corresponds to features for a single CG site,
    except for 'names', which may have names which are shared for each cg site
    (or None).
    """
    # prep information needed for featurization

    # mapped CG points
    cg_points = jnp.asarray(cmap(np.asarray(points)))
    reduced_cons = reduce_constraint_sets(constraints)
    ids = tuple(id_feat(points, cmap, constraints, return_ids=True))
    # matrix for smearing points for constraints
    smearm = jnp.asarray(
        smear_map(
            site_groups=reduced_cons,
            n_sites=cmap.n_fg_sites,
            return_mapping_matrix=True,
        )
    )
    max_channels = max(ids)

    # shared option dict for featurization and div calls
    f_kwargs = {
        "channels": ids,
        "max_channels": max_channels,
        "smear_mat": smearm,
        "inner": inner,
        "outer": outer,
        "width": width,
        "n_basis": n_basis,
        "dist_power": dist_power,
    }

    # we use abatch to break down computation. In order to do so, we make
    # wrapped callables that take simpler arguments

    def subfeater(arg_inds: jax.Array, arg_cg_site: int) -> jax.Array:
        sub_points = points[arg_inds]
        sub_cg_points = cg_points[arg_inds]
        feat = gb_subfeat(
            points=sub_points,
            cg_points=sub_cg_points[:, arg_cg_site : (arg_cg_site + 1), :],
            **f_kwargs,
        )
        return feat

    # subsetted by abatch in feater and divver to mark where to evaluate
    inds = jnp.arange(len(points))

    # make function which batches the JAX calls to keep memory usage down
    def feater(cg_site: int) -> np.ndarray:
        feat = abatch(
            func=subfeater, arr=inds, arg_cg_site=cg_site, chunk_size=batch_size
        )
        return np.asarray(feat)

    if lazy:
        feats: Iterable = (feater(x) for x in range(cmap.n_cg_sites))
    else:
        feats = [feater(x) for x in range(cmap.n_cg_sites)]

    # now do the same for divergences

    # this function takes a set of indices for subsetting, this makes it
    # compatible with abatch
    def subdivver(arg_inds: jax.Array, arg_cg_site: int) -> jax.Array:
        sub_points = points[arg_inds]
        sub_cg_points = cg_points[arg_inds]
        div = gb_subfeat_jac(
            points=sub_points,
            cg_points=sub_cg_points[:, arg_cg_site : (arg_cg_site + 1), :],
            method=div_method,
            **f_kwargs,
        )
        return div

    # make function which batches the JAX calls to keep memory usage down
    def divver(cg_site: int) -> np.ndarray:
        div = abatch(
            func=subdivver, arr=inds, arg_cg_site=cg_site, chunk_size=batch_size
        )
        return np.asarray(div)

    if lazy:
        divs: Iterable = (divver(x) for x in range(cmap.n_cg_sites))
    else:
        divs = [divver(x) for x in range(cmap.n_cg_sites)]

    return {KNAME_FEATS: feats, KNAME_DIVS: divs, KNAME_NAMES: None}


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


@partial(jax.jit, inline=True, static_argnames=["n_basis"])
def gaussian_dist_basis(
    dists: jax.Array,
    outer: float,
    inner: float = 0,
    n_basis: int = 10,
    width: float = 1.0,
    dist_power: float = 0.5,
    clip: float = 1e-3,
) -> jax.Array:
    """Transform arrays of distances into Gaussian "bins" of distances.

    NOTE: This function applies to JAX arrays.

    NOTE: Grid points are uniformed distributed when taken to the power of
    dist_power.

    NOTE: Distances outside inner/outer are not clipped; these values only
    control grid creation.

    Arguments:
    ---------
    dists (jax.Array):
        Array of distances. Can be any shape.
    outer (positive float):
        Ending distance use when creating grid of Gaussians.
    inner (positive float):
        Starting distance use when creating grid of Gaussians.
    n_basis (positive integer):
        Number of Gaussian "bins" to use.
    width (positive float):
        Width of generated Gaussians: Gaussians are defined as exp(d**2/width),
        where d is the offset distance.
    dist_power (float):
        Grid points are uniformed distributed when taken to the power of
        this argument. In other words, linspace is applied after applying the
        transformation x|->x**dist_power, and then mapped back to the original
        resolution. values<1 concentrate points towards the beginning of the
        interval, values>1 concentrate poitns towards the end of the interval.
    clip (float):
        Passed to clipped_gauss.

    Returns:
    -------
    Array where additional dimensions characterizing bins are applied to the
    (-1,) axis position. For example, if dists is shape (2,2) and n_basis=5
    the output shape is (2,2,5).
    """
    pow_grid_points = jnp.linspace(inner**dist_power, outer**dist_power, n_basis)
    grid_points = pow_grid_points ** (1 / dist_power)
    feats = [
        clipped_gauss(inp=dists, center=o, width=width, clip=clip) for o in grid_points
    ]
    return jnp.stack(feats, axis=-1)


@partial(jax.jit, inline=True)
def clipped_gauss(
    inp: jax.Array, center: float, width: float = 1.0, clip: float = 1e-3
) -> jax.Array:
    """Clipped Gaussian.

    Gaussian output is set to zero below a certain value and shifted to be continuous.

    NOTE: This function applies to JAX arrays.

    Gaussian is first calculated as exp(-((inp-center)/width)**2). Then, all
    values are give a minimum value of clip. Finally, clip is subtracted from all
    values.

    Arguments:
    ---------
    inp (jnp.DeviceArray):
        Input values to be filtered through Gaussian. May be any shape.
    center (float):
        Offset subtracted from values before Gaussian is applied.
    width (float):
        Scaling inside Gaussian exponent.
    clip (float):
        Value at which Gaussian output is set to zero.

    Returns:
    -------
    Array the same shape and size as inp, but transformed through a Gaussian.
    """
    gauss = jnp.exp(-(((inp - center) / width) ** 2))
    if clip is None:
        return gauss
    else:
        return jnp.clip(a=gauss, a_min=clip) - clip


@partial(
    jax.jit, inline=True, static_argnames=["channels", "max_channels", "jac_shape"]
)
def channel_allocate(
    feats: jax.Array, channels: Tuple[int], max_channels: int, jac_shape: bool = False
) -> jax.Array:
    """Transform atom features to one hot versions.

    Features given for each atom correspond to one hot versions that independently apply
    to groups of atoms.

    For example, if a frame has 4 fine-grained sites with features [[a,b,c,d]], it
    could be transformed into:
        [[a, 0, 0, 0]]
        [[0, b, 0, 0]]
        [[0, 0, c, 0]]
        [[0, 0, 0, d]]
    This occurs if the channels of the four sites are 0,1,2,3. However, if the
    channels are 0,1,1,2, then they would be transformed into:
        [[a, 0, 0]]
        [[0, b, 0]]
        [[0, c, 0]]
        [[0, 0, d]]
    Channels typically identify groups of constrained atoms. This is similar to
    a one-hot encoding; as a result, most of the values in the resulting feature
    set are zero.

    Arguments:
    ---------
    feats (jax.Array):
        Array containing the features for each fine-grained site at each frame.  Assumed
        to be of shape (n_frames, n_fg_sites, n_feats) or
        (n_feats, n_frames, n_fg_sites, n_dim) (see jac_shape).
    channels (tuple of positive integers):
        Tuple of integers with the length being the number of fine-grained sites
        in the trajectory. Each integer assigns a fine-grained site to a
        constraint group. So, if two atoms have a constrained bond connecting
        them, they should both have the same integer. The integers do not have
        to be consecutive, but max_channels must as big as the largest channel.
    max_channels (positive integer):
        Maximum value of channels. Included as argument due to JAX constraints.
        Larger values increase memory usage, so the most memory efficient
        (max_channels,channels) pair has channels starting at 0 with maximum value
        at max_channels, with no unused index in between.
    jac_shape (boolean):
        If True, feats is assumed to be of shaped (n_feats, n_frames,
        n_fg_sites, n_dim).  Else, feats is assumed to be of shape
        (n_frames, n_fg_sites, n_feats)


    Returns:
    -------
    jax.Array of similar shape as input, but with feats dimension scaled
    to be max_channels*max_feats.
    """
    if jac_shape:
        # jac is (n_feat, n_frame, n_fg_sites, n_dim) noqa false ERA001 trigger
        n_feats = feats.shape[0]
        n_frames = feats.shape[1]
        n_dim = feats.shape[3]

        per_site_arrays = []

        # zero array that each slice in loop is based on
        per_atom_features_base = jnp.zeros((n_feats * max_channels, n_frames, n_dim))
        for site, channel in enumerate(channels):
            # location of particular slice
            target = slice(n_feats * channel, n_feats * (channel + 1))
            # JAX modifications are not in-place
            per_atom_feats = per_atom_features_base.at[target, :, :].set(
                feats[:, :, site, :]
            )
            per_site_arrays.append(per_atom_feats)

        return jnp.stack(per_site_arrays, 2)
    else:
        n_feats = feats.shape[2]
        n_frames = feats.shape[0]

        per_site_arrays = []

        # zero array that each slice in loop is based on
        per_atom_features_base = jnp.zeros((n_frames, n_feats * max_channels))
        for site, channel in enumerate(channels):
            # location of particular slice
            target = slice(n_feats * channel, n_feats * (channel + 1))
            # JAX modifications are not in-place
            per_atom_feats = per_atom_features_base.at[:, target].set(feats[:, site, :])
            per_site_arrays.append(per_atom_feats)
        return jnp.stack(per_site_arrays, 1)


@partial(
    jax.jit,
    static_argnames=[
        "inner",
        "outer",
        "channels",
        "max_channels",
        "collapse",
        "channelize",
        "n_basis",
    ],
)
def gb_subfeat(
    points: jax.Array,
    cg_points: jax.Array,
    channels: Tuple[int],
    max_channels: int,
    smear_mat: Union[None, jax.Array],
    collapse: bool = False,
    channelize: bool = True,
    **kwargs,
) -> jax.Array:
    """Create features (without divergences) using Gaussian bins and distances.

    Points are mapped using smear_mat, per frame distances are calculated,
    these distances are expressed using Gaussian bins, and these bins are then
    distributed over an array to make them type specific.

    Arguments:
    ---------
    points (jax.Array):
        Positions of the fine_grained trajectory. Assumed to have shape
        (n_frames,n_fg_sites,n_dims) or (n_fg_sites,n_dims); in the
        latter case, a dummy n_frames index is added during computation.
    cg_points (jax.Array):
        Positions of coarse-grained trajectory. Assumed to have shape
        (n_frames,n_cg_sites,n_dims). Current usage only considers 1 cg site at a
        time.
    channels (tuple of positive integers):
        Tuple of integers with the length of the number of fine-grained sites
        in the trajectory. Each integer assigns that fine-grained site to a
        constraint group. So, if two atoms have a constrained bond connecting
        them, they should both have the same integer. The integers do not have
        to be consecutive, but max_channels must as big as the largest channel.
    max_channels (positive integer):
        Maximum value of channels. Included as argument due to JAX constraints.
        Larger values increase memory usage, so the most memory efficient
        (max_channels,channels) pair has channels starting at 0 with maximum value
        at max_channels, with no unused index in between.
    smear_mat (jax.Array):
        Mapping matrix multiplied with points via trjdot prior to calculating
        distances. Useful for accounting for molecular constraints. Should be
        shape (n_fg_sites,n_fg_sites).
    collapse (boolean):
        Trace over indices corresponding to frames and fine-grained sites in the
        output. Useful for some later gradient calculations.  If collapse=True
        and points is 2-dimensional, the output may not make sense.
    channelize (boolean):
        Whether to distribute the Gaussian features over one-hot-like channels
        to make them specific to various groups of atoms.
    **kwargs:
        Passed to gaussian_dist_basis.

    Returns:
    -------
    If collapse, an array of shape (n_features,) is returned; else,
    jnp.DeviceArray of ether shape (n_frames,n_fg_sites,n_features) or
    (n_fg_sites,n_features) is returned, with the latter occurring when points
    only has two dimensions.  n_features is set via kwargs, max_channels,
    and gaussian_dist_basis.
    """
    # if our input has no frame axis, add dummy
    if len(points.shape) == 2:
        points = points[None, ...]
        dummy_axis = True
    else:
        dummy_axis = False

    if smear_mat is not None:
        points = trjdot(points, smear_mat)
    dists = distances(xyz=points, cross_xyz=cg_points)
    gauss = gaussian_dist_basis(dists, **kwargs)[:, 0, :, :]
    if channelize:
        channelized = channel_allocate(gauss, channels, max_channels)
    else:
        channelized = gauss
    if collapse:
        collapsed = channelized.sum(axis=(0, 1))
    else:
        collapsed = channelized
        # if we collapse, then this index removal doesn't make sense
        if dummy_axis:
            return collapsed[0, ...]
    return collapsed


def gb_subfeat_jac(
    points: jax.Array,
    cg_points: jax.Array,
    channels: int,
    max_channels: int,
    smear_mat: Union[jax.Array, None] = None,
    method: str = DIVMETHOD_REORDER,
    **kwargs,
) -> jax.Array:
    """Calculate per frame (collapsed) divergences for gb_subfeat.

    Most arguments are passed to gb_subfeat; see that function for more details.
    However, note that not all the arguments are the same (see, for example, the
    allowed shaped of points and where kwargs goes).

    NOTE: Be sure to pass the same arguments to this and gb_subfeat if using
    their results in tandem (even if this function;s internal call to gb_subfeat
    changes certain arguments).

    Arguments:
    ---------
    points (jax.Array):
        Positions of the fine_grained trajectory. Assumed to have shape
        (n_frames,n_fg_sites,n_dims).
    cg_points (jax.Array):
        Positions of coarse-grained trajectory. Assumed to have shape
        (n_frames,n_cg_sites,n_dims); current usage only considers 1 cg site at a
        time.
    channels (tuple of positive integers):
        Tuple of integers with the length of the number of fine-grained sites
        in the trajectory. Each integer assigns that fine-grained site to a
        constraint group. So, if two atoms have a constrained bond connecting
        them, they should both have the same integer. The integers to not have
        to be consecutive, but max_channels must as big as the largest channel.
    max_channels (positive integer):
        Maximum value of channels. Included as argument due to JAX constraints.
        Larger values increase memory usage, so the most memory efficient
        (max_channels,channels) pair has channels starting at 0 with maximum value
        at max_channels, with no unused index in between.
    smear_mat (jax.Array):
        Mapping matrix multiple with points via trjdot prior to calculating
        distances. Useful for accounting for molecular constraints. Should be
        shape (n_fg_sites,n_fg_sites).
    method (string):
        if method=="basic":
            A direct Jacobian is calculated using a full gb_subfeat call with
            collapse=True.
        elif method=="reorder":
            Jacobian is calculated before one-hot-like vectors are created, and
            then itself one-hotted.
    kwargs:
        Passed to gb_subfeat.

    Returns:
    -------
    jax.Array of shape (n_frames, n_features, n_dims=3) containing the per
    frame Jacobian values summed over the fine grained particles.
    """
    if method == DIVMETHOD_BASIC:
        # collapse=True-> sums features over all atoms and frames to that
        # jacobian calculation avoids trivial zero entries.
        def to_jac(x: jax.Array) -> jax.Array:
            return gb_subfeat(
                x,
                cg_points=cg_points,
                channels=channels,
                max_channels=max_channels,
                smear_mat=smear_mat,
                collapse=True,
                **kwargs,
            )

        jac = jax.jacfwd(to_jac)(points)
        # sum over fine-grained sites
        traced_jac = jac.sum(axis=(2,))
        reshaped_jac = jnp.swapaxes(traced_jac, 0, 1)
        return reshaped_jac
    elif method == DIVMETHOD_REORDER:

        def to_jac(x: jax.Array) -> jax.Array:
            return gb_subfeat(
                x,
                cg_points=cg_points,
                channels=channels,
                max_channels=max_channels,
                smear_mat=smear_mat,
                collapse=True,
                channelize=False,
                **kwargs,
            )

        # jac is (n_feat, n_frame, n_fg_sites, n_dim) noqa false ERA001 trigger
        jac = jax.jacrev(to_jac)(points)
        # ch_jac is (exp_n_feat, n_frame, n_fg_sites, n_dim) noqa false ERA001 trigger
        ch_jac = channel_allocate(jac, channels, max_channels, jac_shape=True)
        # sum over fine-grained sites
        traced_ch_jac = ch_jac.sum(axis=(2,))
        reshaped_jac = jnp.swapaxes(traced_ch_jac, 0, 1)
        return reshaped_jac
    else:
        raise ValueError("Unknown method for jacobian calculation.")


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
