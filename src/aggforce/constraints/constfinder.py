"""Tools for inferring constrained bonds from molecular trajectories.

Useful for automatically obtaining a list of molecularly constrained atoms to
feed into mapping methods. Currently, only pairwise distance constraints are
considered.
"""

import numpy as np
from ..util import distances
from .hints import Constraints


def guess_pairwise_constraints(xyz: np.ndarray, threshold: float = 1e-3) -> Constraints:
    """Find pairs of sites which are likely constrained via fluctuations.

    Fluctuations are found by
    The pairwise distances for each frame are calculated; then, the standard
    deviation for each distance over time is calculated. If this standard
    deviation is lower than a threshold, the two atoms are considered
    constrained.

    Arguments:
    ---------
    xyz (numpy.ndarray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    threshold (positive float):
        Distances with standard deviations lower than this value are considered
        to be constrainted. Has units of xyz.

    Returns:
    -------
    A set of frozen sets, each of which contains a pair of indices of sites
    which are guessed to be pairwise constrained.
    """
    dists = distances(xyz)
    sds = np.sqrt(np.var(dists, axis=0))
    np.fill_diagonal(sds, threshold * 2)
    inds = np.nonzero(sds < threshold)
    return {frozenset(v) for v in zip(*inds)}
