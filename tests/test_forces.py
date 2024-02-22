r"""Tests force aggregation schemes using a chignolin (CLN025) trajectory.

Some of the tests in this module require JAX to be installed. They are marked via pytest
decorators.  

We do not have a ground truth for validating these results. The tests in this
module instead use either previously generated values (e.g., checking the map
matrices themselves), or compare two sets of mapped forces for consistency.

The provided tests are sound when considering integrals over the canonical
distribution characteristic to the provided trajectory.  However, these
comparison are subject to statistical and numerical noise, and so are compared
up to numerical thresholds. These thresholds have been decided by existing
results and visual inspection, and are not rigorous in any sense. They only
serve as sanity checks for force generation methods, and should not be regarded
as proof that a force aggregation method is correct.

Additionally, note that the provided trajectory data is not close to converged,
which may lead to additional bias.

In practice, the tests seem to be able to detect if molecular constraints are not
obeyed, but cannot detect if a wrong temperature is specified for featurized
maps.

To adapt these tests to other molecules or sources of data, modify get_data.

The tests in this section make references to coarse-graining (CG) terminology
and ideas. For an introduction see the following publications:

Noid, William George, et al. "The multiscale coarse-graining method. I. A
rigorous bridge between atomistic and coarse-grained models." The Journal of
chemical physics 128.24 (2008): 244114.

Rudzinski, Joseph F., and W. G. Noid. "Coarse-graining entropy, forces, and
structures." The Journal of chemical physics 135.21 (2011): 214101.
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
    constraint_aware_uni_map,
)


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


def test_cln025_basic_agg_forces_against_ref() -> None:
    r"""Test CLN025 basic map formation.

    Test if basic force aggregation produces a map which is the same as
    as a saved map previously generated using the same method.
    """
    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    # make force map
    basic_results = project_forces(
        xyz=coords,
        forces=forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        method=constraint_aware_uni_map,
    )

    location = Path(__file__).parent
    mapfile = str(location / "data/cln_basic_force_mat.txt")
    ref = np.loadtxt(mapfile)
    assert ((basic_results["map"].standard_matrix - ref) ** 2).sum() < 1e-5


def test_cln025_opt_agg_forces_against_ref() -> None:
    r"""Check optimized CLN025 linear force mapping.

    Checks if configuration-independent optimized force aggregation produces a map which
    the same as as a saved map previously made using the same method.
    """
    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    # make force map
    optim_results = project_forces(
        xyz=coords,
        forces=forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        l2_regularization=1,
    )

    location = Path(__file__).parent
    mapfile = str(location / "data/cln_opt_force_mat.txt")
    ref = np.loadtxt(mapfile)
    assert ((optim_results["map"].standard_matrix - ref) ** 2).mean() < 1e-3


@pytest.mark.jax
def test_cln025_opt_basic_rsqpg_mscg_ip(seed: int = rseed) -> None:
    r"""Check if CLN025 basic and optimized forces consistent via projections.

    Checks to see if the force projections along a random generated set of
    basis vectors approximately matches between a basic aggregating force map vs
    a configuration-independent optimized force map.

    This test has typically been sensitive enough to detect if molecular
    constraints are misapplied in either force map.

    This test is based on the following relationship:
        \int F(x) \dot G \circ M (x) = H
    where F(x) is the force map being applied to configuration x, G is a
    function of CG (mapped) coordinates, M is the configurational CG map, and
    the integral is performed over the fine-grained configurational ensemble. G
    is a function that has the same shaped output as the force map, and \dot
    multiplies all values of F and G element-wise ans sums them at each
    configuration.

    Critically, H is fixed for all "correct" F (F with the same conditional
    mean). This implies that two force maps which both are noisy versions of the
    same manybody-PMF will have the same H (in the infinite sample limit).

    We use this relationship by applying the same formula to a finite trajectory
    chunk. Multiple G are used: each is the force output of a randomly CG
    potential that applies a single fixed Gaussian to every member of a
    per-frame distance matrix.

    The coefficients do not match perfectly; we attribute this to using a finite
    trajectory. This test characterizes the statistical properties of the
    two projections to see if they are just noisy versions of each other (we
    test the correlation and the average difference in the projections).

    This test requires JAX since our strategy for G uses JAX's
    autodifferentiation.
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

    basic_results = project_forces(
        xyz=train_coords,
        forces=train_forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        method=constraint_aware_uni_map,
    )

    optim_results = project_forces(
        xyz=train_coords,
        forces=train_forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        l2_regularization=1e3,
    )

    optim_forces = optim_results["map"](test_forces)
    basic_forces = basic_results["map"](test_forces)

    cg_coords = cmap(test_coords)

    kwargs = {
        "coords": cg_coords,
        "n_samples": 1000,
        "inner": 6.0,
        "outer": 12.0,
        "width": 0.5,
        "average": False,
    }

    # mypy gets angry at this kwargs usage, we suppress
    basic_proj = mv.random_force_proj(  # type: ignore [call-overload]
        forces=basic_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )
    optim_proj = mv.random_force_proj(  # type: ignore [call-overload]
        forces=optim_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )

    # check to see if the force signals are highly correlated. This depends on
    # the noise is each force proj, and here is often quite low.
    assert np.abs(np.corrcoef(np.array([basic_proj, optim_proj]))[0, 1]) > 0.5
    diffs = [b - o for b, o in zip(basic_proj, optim_proj)]
    mean_diff = sum(diffs) / len(diffs)
    rmsd_bp = (sum([d**2 for d in basic_proj]) / len(basic_proj)) ** 0.5
    rmsd_op = (sum([d**2 for d in optim_proj]) / len(optim_proj)) ** 0.5
    # check to see if the average relative difference between force estimations
    # is close to zero
    assert np.abs(2 * (mean_diff) / (rmsd_bp + rmsd_op)) < 0.08


@pytest.mark.jax
def test_cln025_opt_basic_rsqpg_offset(seed: int = rseed) -> None:
    r"""Test if optimized and basic forces have similar residual differences.

    Checks to see if the difference in force residual between two force-fields is the
    same for a force map based on simple aggregation and that created using
    configuration-independent optimized force.

    This is based on the following property. The configurational integral form
    of the force residual (here denoted R) evaluated for force-field G has a
    noise term and a noiseless term as so:
        R[G] = Noise + R_noiseless[G]
    The noise term does not depend on G. As a result, if you take the difference
    between R for two force-fields they cancel out:
        R[G_0] - R[G_1] = R_noiseless[G_0] - R_noiseless[G_1]
    Critically, the choice of force-map only affects the Noise term. In other
    words, if we take the difference in two force residuals, the noise does not
    matter.

    Note that this logic is developed using the force-residual created from a
    configurational integral (i.e., in the infinite sampling limit). The
    properties found on finite trajectories may differ.

    We use this relationship by applying the same formula to a finite trajectory
    chunk. Multiple G are used:the force output of a randomly CG
    potential that applies a single Gaussian to every member of a per-frame
    distance matrix is compared to a constant force-field that has 0 force
    everywhere.

    The coefficients do not match perfectly; we attribute this to using a finite
    trajectory. This test characterizes the statistical properties of the
    calculated offsets to see if they are just noisy versions of each other (we
    test the correlation and the average difference in the projections).

    This test requires JAX since our strategy for G uses JAX's
    autodifferentiation.
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

    basic_results = project_forces(
        xyz=train_coords,
        forces=train_forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        method=constraint_aware_uni_map,
    )

    optim_results = project_forces(
        xyz=train_coords,
        forces=train_forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        l2_regularization=1e3,
    )

    optim_forces = optim_results["map"](test_forces)
    basic_forces = basic_results["map"](test_forces)

    cg_coords = cmap(test_coords)

    kwargs = {
        "coords": cg_coords,
        "n_samples": 1000,
        "inner": 6.0,
        "outer": 12.0,
        "width": 0.5,
        "average": False,
    }

    basic_proj = mv.random_residual_shift(  # type: ignore [call-overload]
        forces=basic_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )
    optim_proj = mv.random_residual_shift(  # type: ignore [call-overload]
        forces=optim_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )

    # unlike previous tests, the lowered noise makes this correlation high
    assert np.abs(np.corrcoef(np.array([basic_proj, optim_proj]))[0, 1]) > 0.99
    diffs = [b - o for b, o in zip(basic_proj, optim_proj)]
    mean_diff = sum(diffs) / len(diffs)
    rmsd_bp = (sum([d**2 for d in basic_proj]) / len(basic_proj)) ** 0.5
    rmsd_op = (sum([d**2 for d in optim_proj]) / len(optim_proj)) ** 0.5
    assert np.abs((2 * (mean_diff) / (rmsd_bp + rmsd_op))) < 0.0025


@pytest.mark.slow
@pytest.mark.jax
def test_cln025_featopt_opt_rsqpg_mscg_ip(seed: int = rseed) -> None:
    r"""Check CLN025 optimized nonlinear and linear forces via projections.

    Checks to see if the force projections along a random generated set of basis vectors
    approximately matches between a basic aggregating force map vs a
    configuration-independent optimized force map.

    See test_cln025_opt_basic_rsqpg_mscg_ip for more details. This is the same
    test, but is applied to the feature-optimized and configuration-independent
    optimized maps.

    This test can be a bit slower since the optimization is more extensive. The
    featurized map is quite basic as we do not test on enough data to see even a
    marginal gain in the force score.
    """
    from aggforce import jaxmapval as mv
    from aggforce.qp import (
        Multifeaturize,
        gb_feat,
        GeneralizedFeatures,
        id_feat,
        qp_feat_linear_map,
    )
    from aggforce.util import Curry

    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    N_FRAMES: Final = 500

    train_coords = coords[:N_FRAMES]
    test_coords = coords[N_FRAMES:]

    train_forces = forces[:N_FRAMES]
    test_forces = forces[N_FRAMES:]

    basic_optim_results = project_forces(
        xyz=train_coords,
        forces=train_forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        l2_regularization=1e3,
    )

    f0: Curry[GeneralizedFeatures] = Curry(
        gb_feat,
        inner=0.0,
        outer=8.0,
        width=1.0,
        n_basis=4,
        lazy=True,
    )
    comb_f = Multifeaturize([id_feat, f0])  # type: ignore [list-item]
    optim_results = project_forces(
        xyz=coords,
        forces=forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        featurizer=comb_f,
        method=qp_feat_linear_map,
        kbt=kbt,
        l2_regularization=1e4,
    )

    cg_coords = cmap(test_coords)

    optim_forces = optim_results["map"](points=test_forces, copoints=test_coords)
    basic_forces = basic_optim_results["map"](points=test_forces, copoints=test_coords)

    kwargs = {
        "coords": cg_coords,
        "n_samples": 1000,
        "inner": 6.0,
        "outer": 12.0,
        "width": 0.5,
        "average": False,
    }

    # the kwargs usage here is incompatible with mypy inference
    basic_optim_proj = mv.random_force_proj(  # type: ignore [call-overload]
        forces=basic_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )
    optim_proj = mv.random_force_proj(  # type: ignore [call-overload]
        forces=optim_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )

    # unlike previous tests, the lowered noise makes this correlation high
    assert np.abs(np.corrcoef(np.array([basic_optim_proj, optim_proj]))[0, 1]) > 0.95
    diffs = [b - o for b, o in zip(basic_optim_proj, optim_proj)]
    mean_diff = np.abs(sum(diffs) / len(diffs))
    rmsd_bp = (sum([d**2 for d in basic_optim_proj]) / len(basic_optim_proj)) ** 0.5
    rmsd_op = (sum([d**2 for d in optim_proj]) / len(optim_proj)) ** 0.5
    assert np.abs((2 * (mean_diff) / (rmsd_bp + rmsd_op))) < 0.01


@pytest.mark.slow
@pytest.mark.jax
def test_cln025_featopt_opt_rsqpg_offset(seed: int = rseed) -> None:
    r"""Test if optimized linear and nonlinear forces have similar residual differences.

    Checks to see if the difference in force residual between two
    force-fields is the same for a force map based on simple aggregation and
    that created using configuration-independent optimized force.

    See test_cln025_opt_basic_rsqpg_offset for more details.  This is the same
    test, but is applied to the feature-optimized and configuration-independent
    optimized maps.

    This test can be a bit slower since the optimization is more extensive. The
    featurized map is quite basic as we do not test on enough data to see even a
    marginal gain in the force score.
    """
    from aggforce import jaxmapval as mv
    from aggforce.qp import (
        Multifeaturize,
        gb_feat,
        id_feat,
        qp_feat_linear_map,
    )
    from aggforce.util import Curry

    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    train_coords = coords[:500]
    test_coords = coords[500:]

    train_forces = forces[:500]
    test_forces = forces[500:]

    basic_opt_results = project_forces(
        xyz=train_coords,
        forces=train_forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        method=constraint_aware_uni_map,
    )
    f0 = Curry(
        gb_feat,
        inner=0.0,
        outer=8.0,
        width=1.0,
        n_basis=4,
        lazy=True,
    )
    comb_f = Multifeaturize([id_feat, f0])  # type: ignore [list-item]
    optim_results = project_forces(
        xyz=coords,
        forces=forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        featurizer=comb_f,
        method=qp_feat_linear_map,
        kbt=kbt,
        l2_regularization=1e4,
    )

    optim_forces = optim_results["map"](points=test_forces, copoints=test_coords)
    basic_opt_forces = basic_opt_results["map"](
        points=test_forces, copoints=test_coords
    )

    cg_coords = cmap(test_coords)

    kwargs = {
        "coords": cg_coords,
        "n_samples": 1000,
        "inner": 6.0,
        "outer": 12.0,
        "width": 0.5,
        "average": False,
    }

    basic_opt_proj = mv.random_residual_shift(  # type: ignore [call-overload]
        forces=basic_opt_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )
    optim_proj = mv.random_residual_shift(  # type: ignore [call-overload]
        forces=optim_forces,
        randg=r.default_rng(seed=seed),
        **kwargs,
    )

    # unlike previous tests, the lowered noise makes this correlation high
    assert np.abs(np.corrcoef(np.array([basic_opt_proj, optim_proj]))[0, 1]) > 0.99
    diffs = [b - o for b, o in zip(basic_opt_proj, optim_proj)]
    mean_diff = sum(diffs) / len(diffs)
    rmsd_bp = (sum([d**2 for d in basic_opt_proj]) / len(basic_opt_proj)) ** 0.5
    rmsd_op = (sum([d**2 for d in optim_proj]) / len(optim_proj)) ** 0.5
    assert np.abs((2 * (mean_diff) / (rmsd_bp + rmsd_op))) < 0.0025
