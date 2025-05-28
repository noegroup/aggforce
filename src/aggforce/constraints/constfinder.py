"""Tools for inferring constrained bonds from molecular trajectories.

Useful for automatically obtaining a list of molecularly constrained atoms to
feed into mapping methods. Currently, only pairwise distance constraints are
considered.
"""

import numpy as np
from ..util import distances, chunker
from .hints import Constraints


def guess_pairwise_constraints(
    xyz: np.ndarray, threshold: float = 1e-3, n_batches: int = 1
) -> Constraints:
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
    n_batches (int):
        number of baches over which to divide the number of sites. As the
        constraint finder has O(n_sites^2) memory requirement, large systems
        will require a lot of RAM, so batching is required for low memory systems

    Returns:
    -------
    A set of frozen sets, each of which contains a pair of indices of sites
    which are guessed to be pairwise constrained.
    """
    if n_batches == 1:
        dists = distances(xyz)
        sds = np.sqrt(np.var(dists, axis=0))
        np.fill_diagonal(sds, threshold * 2)
        inds = np.nonzero(sds < threshold)
        return {frozenset(v) for v in zip(*inds)}
    else:
        n_sites = xyz.shape[1]
        elem_chunks = chunker(np.arange(n_sites), n_batches)
        constraints = set()
        for i, first_entry_chunk in enumerate(elem_chunks):
            for j, second_entry_chunk in enumerate(elem_chunks):
                displacement_matrix = (
                    xyz[:, None, first_entry_chunk, :]
                    - xyz[:, second_entry_chunk, None, :]
                )
                distance_matrix = np.linalg.norm(displacement_matrix, axis=-1)
                sds = np.sqrt(np.var(distance_matrix, axis=0))
                if i == j:
                    np.fill_diagonal(sds, threshold * 2)
                inds = np.nonzero(sds < threshold)
                if len(inds[0]) > 0 and len(inds[1]) > 0:
                    local_constraints = {
                        frozenset(v)
                        for v in zip(
                            second_entry_chunk[inds[0]], first_entry_chunk[inds[1]]
                        )
                    }
                    constraints = constraints.union(local_constraints)
        return constraints
