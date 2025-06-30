"""Tools for inferring constrained bonds from molecular trajectories.

Useful for automatically obtaining a list of molecularly constrained atoms to
feed into mapping methods. Currently, only pairwise distance constraints are
considered.
"""

import numpy as np
from ..util import distances
from .hints import Constraints
from typing import Union


def guess_pairwise_constraints(
    xyz: np.ndarray, cross_xyz: Union[None, np.ndarray] = None, threshold: float = 1e-3
) -> Constraints:
    """
    Find pairs of sites which are likely constrained via fluctuations.

    Fluctuations are estimated by computing the pairwise distances for each frame,
    then evaluating the standard deviation of those distances over time. Pairs
    with low fluctuation are assumed constrained.

    Arguments
    ---------
    xyz : numpy.ndarray
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps, n_sites, n_dim).
    cross_xyz : np.ndarray or None, optional
        Array of shape (n_steps, other_n_sites, n_dim) to compare distances between
        xyz and another group of atoms. If None, distances are computed within xyz.
    threshold : positive float
        Distances with standard deviations lower than this value are considered
        to be constrained. Has units of xyz.

    Returns
    -------
    set
        - If cross_xyz is None: set of frozensets of symmetric index pairs (i, j),
          each of which contains a pair of indices of sites which are guessed to be
          pairwise constrained.
        - If cross_xyz is provided: set of ordered tuples (i, j) where i indexes
          cross_xyz (other_n_sites) and j indexes xyz (n_sites): this is done for compatibility with the internally called distances function.
            The frozenset is not used here because the order of the sites matters when comparing two different systems.
    """
    dists = distances(xyz, cross_xyz=cross_xyz)
    sds = np.sqrt(np.var(dists, axis=0))

    if cross_xyz is None:
        # Avoid counting self-pairs as constrained
        np.fill_diagonal(sds, threshold * 2)
        inds = np.nonzero(sds < threshold)
        return {frozenset(v) for v in zip(*inds)}
    else:
        # There are no self-pairs in cross_xyz, so we can use ordered pairs
        inds = np.nonzero(sds < threshold)
        return {(i, j) for i, j in zip(*inds)}
