r"""Tests LinearMap behavior for correctness.

Many tests in this module require JAX to be installed. They are marked via pytest 
decorators.
"""
from typing import Final, Any
import numpy as np
from numpy import float32
import pytest
from aggforce import (
    LinearMap,
)

N_FG_SITES: Final = 15
N_FG_FRAMES: Final = 20
N_CG_SITES: Final = 5
N_DIM: Final = 3

JAXNP_TOL: Final = 1e-6

TOL: Final = 1e-4
FINE_TOL: Final = 1e-12

# this seeds some portions of the randomness of these tests, but not be
# complete.
rseed: Final = 42100


@pytest.fixture
def random_fg_positions(seed: int = rseed) -> np.ndarray:
    """Generate random array representing a trajectory fine-grained positions."""
    rng = np.random.default_rng(seed=seed)
    return 100 * (rng.random(size=(N_FG_FRAMES, N_FG_SITES, N_DIM)) - 0.5)


@pytest.fixture
def random_cgmap_matrix(seed: int = rseed) -> np.ndarray:
    """Generate random array representing dense coarse-grained mapping.

    Note that this is _not_ a slice map, and probably scales the data.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.random(size=(N_CG_SITES, N_FG_SITES))


# basic l2 distance
def _l2(array1: Any, array2: Any, /, mean: bool = False) -> float:
    diff = (((array1 - array2) ** 2).sum()) ** (0.5)
    if mean:
        return diff / array1.size
    else:
        return diff


@pytest.mark.jax
def test_jlinearmap(
    random_fg_positions: np.ndarray, random_cgmap_matrix: np.ndarray
) -> None:
    r"""Test if JLinearMap and LinearMap are consistent with each other.

    Tests output types, numerical consistency, and flat_call behavior. Note that
    JAX and numpy seem to differ in the exact numerical output.
    """
    from aggforce.map import JLinearMap
    import jax.numpy as jnp
    import jax

    lmap = LinearMap(mapping=random_cgmap_matrix)
    jlmap = JLinearMap(mapping=random_cgmap_matrix)

    # jlmap should act on nd arrays to produce np.ndarrays
    assert isinstance(jlmap(random_fg_positions), np.ndarray)

    # the output should match
    difference = _l2(jlmap(random_fg_positions), lmap(random_fg_positions), mean=True)
    assert difference < JAXNP_TOL

    #
    derived_jlmap = JLinearMap.from_linearmap(lmap)
    difference = _l2(
        derived_jlmap(random_fg_positions), lmap(random_fg_positions), mean=True
    )
    assert difference < JAXNP_TOL

    # jlmap should act on jax.Arrays to produce jax.Arrays
    j_random_fg_positions = jnp.asarray(random_fg_positions)
    assert isinstance(jlmap(j_random_fg_positions), jax.Array)

    # the produced jax array should match the numpy array output
    difference = _l2(
        jlmap(random_fg_positions), jlmap(j_random_fg_positions), mean=True
    )
    assert difference < JAXNP_TOL

    flattened = random_fg_positions.reshape(N_FG_FRAMES, N_FG_SITES * N_DIM)
    difference = _l2(jlmap.flat_call(flattened), lmap.flat_call(flattened), mean=True)
    assert difference < JAXNP_TOL


def test_flat_linearmap(
    random_fg_positions: np.ndarray, random_cgmap_matrix: np.ndarray
) -> None:
    r"""Test if LinearMap.flat_call acts as expected in comparison to __call__."""
    lmap = LinearMap(mapping=random_cgmap_matrix)
    flattened = random_fg_positions.reshape(
        random_fg_positions.shape[0],
        random_fg_positions.shape[1] * random_fg_positions.shape[2],
    )
    normal_mapped = lmap(random_fg_positions)
    postflattened_mapped = normal_mapped.reshape(
        normal_mapped.shape[0],
        normal_mapped.shape[1] * normal_mapped.shape[2],
    )
    preflattened_mapped = lmap.flat_call(flattened)
    assert np.allclose(preflattened_mapped, postflattened_mapped)


def test_linearmap_precision_direct(random_cgmap_matrix: np.ndarray) -> None:
    r"""Test downgrading precision of LinearMap instances.

    Compares with manual casting.

    """
    lmap = LinearMap(mapping=random_cgmap_matrix)

    difference = _l2(
        lmap.astype(float32).standard_matrix, lmap.standard_matrix.astype(float32)
    )

    assert difference < FINE_TOL


def test_linearmap_precision_mapping(
    random_fg_positions: np.ndarray, random_cgmap_matrix: np.ndarray
) -> None:
    r"""Test downgrading precision of LinearMap instances.

    Compares with mapped output of float64 LinearMap.

    """
    lmap = LinearMap(mapping=random_cgmap_matrix)
    lmap_32 = lmap.astype(float32)
    random_fg_positions_32 = random_fg_positions.astype(float32)

    mapped_32 = lmap_32(random_fg_positions_32)
    mapped = lmap(random_fg_positions_32)

    difference = _l2(mapped_32, mapped)

    assert difference < TOL


@pytest.mark.jax
def test_jlinearmap_precision_mapping(
    random_fg_positions: np.ndarray, random_cgmap_matrix: np.ndarray
) -> None:
    r"""Test downgrading precision of JLinearMap instances.

    Compares with LinearMap results.

    """
    from aggforce.map import JLinearMap

    lmap = LinearMap(mapping=random_cgmap_matrix)
    jlmap = JLinearMap(mapping=random_cgmap_matrix)

    lmap_32 = lmap.astype(float32)
    jlmap_32 = jlmap.astype(float32)

    random_fg_positions_32 = random_fg_positions.astype(float32)

    mapped_32 = lmap_32(random_fg_positions_32)
    jmapped_32 = jlmap_32(random_fg_positions_32)

    difference = _l2(mapped_32, jmapped_32)

    assert difference < TOL
