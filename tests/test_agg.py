"""Test linear optimized force map generated for a water dimer.

We expect that an optimal force map for a configurational map isolating the oxygens will
include contributions from the hydrogens. This test confirms that the linear
force map optimization scheme returns this result.

No bond constraints are present in the reference trajectory.

This result is not a mathematical certainty, but has been empirically true.
"""
from pathlib import Path
import numpy as np
from aggforce import LinearMap, project_forces


def test_agg_opt() -> None:
    """Test optimized force aggregation for a flexible water dimer."""
    location = Path(__file__).parent
    dimerfile = str(location / "data/waterdimer.npz")
    dimerdata = np.load(dimerfile)
    forces = dimerdata["Fs"]

    # CG mapping: two oxygens
    inds = [[0], [3]]
    cmap = LinearMap(inds, n_fg_sites=forces.shape[1])
    optim_results = project_forces(
        xyz=None,
        forces=forces,
        config_mapping=cmap,
        constrained_inds=set(),
        solver_args={"solver": "scs"},
    )

    # aggregation mapping: we expect that contributions from each water are added up
    # to cancel the intramolecular bond forces
    agg_mapping = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=float)
    assert np.allclose(optim_results["map"].standard_matrix, agg_mapping, atol=5e-3)
