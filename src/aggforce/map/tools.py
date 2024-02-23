"""Routines for creating specialized maps.

This routines avoid the process of repeatedly creating maps by hand in specific
contexts.
"""
from typing import Union, Iterable, overload, Literal
from itertools import combinations, product
import numpy as np
from ..trajectory import AugmentedTrajectory
from .core import LinearMap


def lmap_augvariables(aug: AugmentedTrajectory) -> LinearMap:
    """Create a LinearMap that isolates added sites in an AugmentedTrajectory.

    AugmentedTrajectory instances have some particles which belong to a physical
    system and some which are from an Augmenter instance. This method detects
    which particles and added through Augmentation and Creates a slice map
    that isolates them.

    Arguments:
    ---------
    aug:
        AugmentedTrajectory instance to derive a map from.

    Returns:
    -------
    LinearMap that operates on the given AugmetedTrajectory instance. Note that this
    is different that an AugmentedTMap, but may be useful for creating one.
    """
    # The particles after the real particles are the augmented particles.
    inds = [[x] for x in range(aug.n_real_sites, aug.n_sites)]
    return LinearMap(inds, n_fg_sites=aug.n_sites)


@overload
def smear_map(
    site_groups: Iterable[Iterable[int]],
    n_sites: int,
    return_mapping_matrix: Literal[True],
) -> np.ndarray:
    ...


@overload
def smear_map(
    site_groups: Iterable[Iterable[int]],
    n_sites: int,
    return_mapping_matrix: Literal[False],
) -> LinearMap:
    ...


@overload
def smear_map(
    site_groups: Iterable[Iterable[int]],
    n_sites: int,
    return_mapping_matrix: Literal[False] = ...,
) -> LinearMap:
    ...


def smear_map(
    site_groups: Iterable[Iterable[int]],
    n_sites: int,
    return_mapping_matrix: bool = False,
) -> Union[LinearMap, np.ndarray]:
    """LinearMap which replaces the groups of atoms with their mean.

    Arguments:
    ---------
    site_groups (list of iterables of integers):
        List of iterables, each member of which describes a group of sites
        which must be "smeared" together.
    n_sites (integer):
        Total number of sites in the system
    return_mapping_matrix (boolean):
        If true, instead of a LinearMap, the mapping matrix itself is returned.

    Returns:
    -------
    LinearMap instance or 2-dimensional numpy.ndarray

    Notes:
    -----
    This map does _not_ reduce the dimensionality of a system;
    instead, every modified position is replaced with the corresponding mean.
    """
    site_sets = [set(x) for x in site_groups]

    for pair in combinations(site_sets, 2):
        if pair[0].intersection(pair[1]):
            raise ValueError(
                "Site definitions in site_groups overlap; merge before passing."
            )

    matrix = np.zeros((n_sites, n_sites), dtype=np.float32)
    np.fill_diagonal(matrix, 1)
    for group in site_sets:
        inds0, inds1 = zip(*product(group, group))
        matrix[inds0, inds1] = 1 / len(group)
    if return_mapping_matrix:
        return matrix
    return LinearMap(mapping=matrix)
