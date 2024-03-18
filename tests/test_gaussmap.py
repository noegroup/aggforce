r"""Tests Gaussian maps using a chignolin (CLN025) trajectory.

This module uses test ideas from test_forces. See that test module for more details.
"""

from typing import Tuple, Final
from pathlib import Path
import re
import numpy as np
import numpy.random as r
import mdtraj as md  # type: ignore [import-untyped]
import pytest
from aggforce import (
    guess_pairwise_constraints,
    LinearMap,
    project_forces,
    joptgauss_map,
    stagedjoptgauss_map,
)
from aggforce.agg import TMAP_KNAME
from aggforce import jaxmapval as mv


# this seeds some portions of the randomness of these tests, but not be
# complete.
rseed: Final = 42100


def get_data() -> Tuple[np.ndarray, np.ndarray, md.Trajectory, float]:
    r"""Return data for tests.

    This is currently grabs a numpy trajectory file, extracts coordinates and forces,
    and then along with a pdb-derived mdtraj trajectory and kbt value returns them.

    Note that we must manually provide a value for KbT in appropriate units.

    Returns
    -------
    A tuple of the following:
        coordinates array
            array of positions as a function of time (shape should be
            (n_frames,n_sites,n_dims)). Should correspond to the same frames
            as the forces array.
        forces array
            array of forces as a function of time (shape should be
            (n_frames,n_sites,n_dims)). Should correspond to the same frames
            as the coordinates array.
        mdtraj.Trajectory
            mdtraj trajectory corresponding to the sites in the coordinates and
            forces array. We use it to make the configurational map by
            considering the atom names, although the method used to generate the
            configurational map may be modified. It does not need more than one
            frame (it can be generated from a pdb).
        KbT (float)
            Boltzmann's constant times the temperature of the reference
            trajectory. See code for units.
    """
    # kbt for 350K in kcal/mol, known a priori for our trajectory files
    kbt = 0.6955215
    location = Path(__file__).parent
    trajfile = str(location / "data/cln025_record_2_prod_97.npz")
    data = np.load(trajfile)
    forces = data["Fs"]
    coords = data["coords"]
    pdbfile = str(location / "data/cln025.pdb")
    pdb = md.load(pdbfile)
    return (coords, forces, pdb, kbt)


def gen_config_map(pdb: md.Trajectory, string: str = "CA$") -> LinearMap:
    r"""Create the configurational map.

    This is needed as it defines constraints which dictate which force maps are
    feasible.

    We here generate a (usually carbon alpha) configurational map using mdtraj's
    topology. The map could also be specified externally.

    Arguments:
    ---------
    pdb (mdtraj.Trajectory):
        Trajectory object describing the fine-grained (e.g. atomistic)
        resolution.
    string (string):
        Regex string which is compared against the str() of the topology.atoms
        entry--- if matched that atom is retained in the configurational map.

    Returns:
    -------
    A LinearMap object which characterizes the configurational map. There are
    multiple ways to initialize this object; see the main code for more details.
    """
    inds = []
    atomlist = list(pdb.topology.atoms)
    # record which atoms match the string via str casing, e.g., which are carbon alphas.
    for ind, a in enumerate(atomlist):
        if re.search(string, str(a)):
            inds.append([ind])
    return LinearMap(inds, n_fg_sites=pdb.xyz.shape[1])


@pytest.mark.jax
def test_cln025_gauss_mscg_ip(seed: int = rseed) -> None:
    r"""Check if CLN025 gauss maps produce known results.

    This checks for consistency against previous results, but not correctness.

    This test is stochastic. It should rarely fail, but it can. We keep it stochastic
    as small design changes may alter seed dependence.

    See tests in test_forces for more information.
    """
    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    train_coords = coords[:500]
    test_coords = coords[500:]

    train_forces = forces[:500]
    test_forces = forces[500:]

    # we do NOT set the rng here.
    gauss_results = project_forces(
        coords=train_coords,
        forces=train_forces,
        coord_map=cmap,
        constrained_inds=constraints,
        l2_regularization=1e3,
        method=joptgauss_map,
        var=0.002,
        kbt=kbt,
    )

    # map multiple times with gauss map to make big generated dataset
    mapped_coords = []
    mapped_forces = []
    for _ in range(300):
        gauss_coords, gauss_forces = gauss_results[TMAP_KNAME].map_arrays(
            test_coords, test_forces
        )
        mapped_coords.append(gauss_coords)
        mapped_forces.append(gauss_coords)

    all_mapped_coords = np.concatenate(mapped_coords, axis=0)
    all_mapped_forces = np.concatenate(mapped_coords, axis=0)

    # project onto random bases. We here give an rng so that we get the same
    # projections.
    gauss_projs = mv.random_force_proj(  # type: ignore
        coords=all_mapped_coords,
        forces=all_mapped_forces,
        randg=r.default_rng(seed=seed),
        n_samples=5,
        inner=6.0,
        outer=12.0,
        width=6.0,
        average=False,
    )
    KNOWN_PROJS: Final = np.array(
        [
            86.73444366455078,
            -87.3666763305664,
            70.80025482177734,
            64.30303955078125,
            -25.622215270996094,
        ]
    )
    assert np.allclose(KNOWN_PROJS, np.array(gauss_projs), atol=2e-1)


@pytest.mark.jax
def test_cln025_sepgauss_mscg_ip(seed: int = rseed) -> None:
    r"""Check if CLN025 seperable gauss maps produce known results.

    This checks for consistency against previous results, but not correctness.

    This test is stochastic. It should rarely fail, but it can. We keep it stochastic
    as small design changes may alter seed dependence.

    See tests in test_forces for more information.
    """
    from aggforce import jaxmapval as mv

    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    train_coords = coords[:500]
    test_coords = coords[500:]

    train_forces = forces[:500]
    test_forces = forces[500:]

    # we do NOT set the rng here.
    gauss_results = project_forces(
        coords=train_coords,
        forces=train_forces,
        coord_map=cmap,
        constrained_inds=constraints,
        premap_l2_regularization=1e3,
        l2_regularization=1e0,
        method=stagedjoptgauss_map,
        var=0.002,
        kbt=kbt,
    )

    # map multiple times with gauss map to make big generated dataset
    mapped_coords = []
    mapped_forces = []
    for _ in range(300):
        gauss_coords, gauss_forces = gauss_results[TMAP_KNAME].map_arrays(
            test_coords, test_forces
        )
        mapped_coords.append(gauss_coords)
        mapped_forces.append(gauss_coords)

    all_mapped_coords = np.concatenate(mapped_coords, axis=0)
    all_mapped_forces = np.concatenate(mapped_coords, axis=0)

    # project onto random bases. We here give an rng so that we get the same
    # projections.
    gauss_projs = mv.random_force_proj(  # type: ignore
        coords=all_mapped_coords,
        forces=all_mapped_forces,
        randg=r.default_rng(seed=seed),
        n_samples=5,
        inner=6.0,
        outer=12.0,
        width=6.0,
        average=False,
    )
    # these were taking from the non seperable gauss test, but also
    # be matched here.
    KNOWN_PROJS: Final = np.array(
        [
            86.73444366455078,
            -87.3666763305664,
            70.80025482177734,
            64.30303955078125,
            -25.622215270996094,
        ]
    )
    assert np.allclose(KNOWN_PROJS, np.array(gauss_projs), atol=2e-1)


@pytest.mark.jax
def test_negative_cln025_sepgauss_mscg_ip(seed: int = rseed) -> None:
    r"""Check if changing parameters causes tests to fail.

    This checks to see if the given tests can detect problems.

    This test is stochastic. It should rarely fail, but it can. We keep it stochastic
    as small design changes may alter seed dependence.

    See tests in test_forces for more information.
    """
    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map

    ################### here we mess with the coordinate map. #####################
    cmap = 2 * (gen_config_map(pdb, "CA$"))
    ################### this is what should make the test fail ####################

    # other operations mirror the other tests in this module

    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    train_coords = coords[:500]
    test_coords = coords[500:]

    train_forces = forces[:500]
    test_forces = forces[500:]

    # we do NOT set the rng here.
    gauss_results = project_forces(
        coords=train_coords,
        forces=train_forces,
        coord_map=cmap,
        constrained_inds=constraints,
        premap_l2_regularization=1e3,
        l2_regularization=1e0,
        method=stagedjoptgauss_map,
        var=0.002,
        kbt=kbt,
    )

    # map multiple times with gauss map to make big generated dataset
    mapped_coords = []
    mapped_forces = []
    for _ in range(300):
        gauss_coords, gauss_forces = gauss_results[TMAP_KNAME].map_arrays(
            test_coords, test_forces
        )
        mapped_coords.append(gauss_coords)
        mapped_forces.append(gauss_coords)

    all_mapped_coords = np.concatenate(mapped_coords, axis=0)
    all_mapped_forces = np.concatenate(mapped_coords, axis=0)

    # project onto random bases. We here give an rng so that we get the same
    # projections.
    gauss_projs = mv.random_force_proj(  # type: ignore
        coords=all_mapped_coords,
        forces=all_mapped_forces,
        randg=r.default_rng(seed=seed),
        n_samples=5,
        inner=6.0,
        outer=12.0,
        width=6.0,
        average=False,
    )

    # these are the values for previous gauss tests. we should _not_ match them.
    KNOWN_PROJS: Final = np.array(
        [
            86.73444366455078,
            -87.3666763305664,
            70.80025482177734,
            64.30303955078125,
            -25.622215270996094,
        ]
    )
    # check that we do not match
    assert not np.allclose(KNOWN_PROJS, np.array(gauss_projs), atol=2e-1)
