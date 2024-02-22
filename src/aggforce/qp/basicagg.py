"""Provides tools for basic aggregation force maps."""

from typing import Union
from ..map import LinearMap
from ..constraints import Constraints, reduce_constraint_sets
from itertools import product
import numpy as np


def constraint_aware_uni_map(
    config_mapping: LinearMap,
    constraints: Union[None, Constraints] = None,
    xyz: Union[None, np.ndarray] = None,  # noqa: ARG001
    forces: Union[None, np.ndarray] = None,  # noqa: ARG001
) -> LinearMap:
    r"""Produce a uniform basic force map compatible with constraints.

    The given configurational map associates various fine-grained (fg) sites with
    each coarse grained (cg) site. This creates a force-map which:
        - aggregates forces from each fg site that contributes to a cg site
        - aggregates forces from atoms which are constrained with atoms included
          via the previous point

    No weighting is applied to the forces before aggregation.

    For example, if we use a carbon alpha slice configurational mapping, any
    carbon alphas which are constrained to hydrogens will have the forces from
    those hydrogens aggregated with their forces. Carbon alphas that are not
    connected to constrained atoms

    NOTE: The configurational map is not checked for any kind of correctness.

    Arguments:
    ---------
    config_mapping (linearmap.LinearMap):
        LinearMap object characterizing the configurational map characterizing
        the connection between the fine-grained and coarse-grained systems.
    constraints (None or set of frozen sets):
        Each set entry is a set of indices, the group of which is constrained.
        Currently, only bond constraints (frozen sets of 2 elements) are supported.
    xyz:
        Ignored. Included for compatibility with other mapping methods.
    forces:
        Ignored. Included for compatibility with other mapping methods.

    Returns:
    -------
    LinearMap object describing a force-mapping.
    """
    if constraints is None:
        constraints = set()
    # get which sites have nonzero contributions to each cg site
    cg_sets = [set(np.nonzero(row)[0]) for row in config_mapping.standard_matrix]
    constraints = reduce_constraint_sets(constraints)
    # add atoms which are related by constraint to those already in cg sites
    for group, x in product(cg_sets, constraints):
        if group.intersection(x):
            group.update(x)
    force_map_mat = np.zeros_like(config_mapping.standard_matrix)
    # place a 1 where all original or those pulled in by constraints are
    for cg_index, cg_contents in enumerate(cg_sets):
        force_map_mat[cg_index, list(cg_contents)] = 1.0
    return LinearMap(force_map_mat)
