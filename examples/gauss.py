"""Demonstrates generating a Gaussian map for Chignolin.

NOTE: You must have JAX and mdtraj installed to run this script.

This is an example script that shows how to load data, find bond constraints,
and create a map that injects Gaussian noise (and modifies the forces). Two
methods are demonstrated. No output or analysis is performed on the produced
maps.
"""

from typing import (
    Tuple,
)
import re
from pathlib import Path
import numpy as np
import mdtraj as md  # type: ignore [import-untyped]

# tools for preparing map optimization call
from aggforce import (
    LinearMap,
    guess_pairwise_constraints,
    joptgauss_map,
    stagedjoptgauss_map,
    project_forces,
)
from aggforce.map import TMap


def get_data() -> Tuple[np.ndarray, np.ndarray, md.Trajectory, float]:
    r"""Return data for analysis.

    This is currently grabs a group of numpy coordinate and force files, stacks them,
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
    kbt = 0.6955215  # kbt for 350K in kcal/mol, known a priori

    force_list = [
        np.load(str(name))["Fs"] for name in Path().glob("record_*_prod_*.npz")
    ]
    coord_list = [
        np.load(str(name))["coords"] for name in Path().glob("record_*_prod_*.npz")
    ]
    forces = np.vstack(force_list)
    coords = np.vstack(coord_list)
    pdb = md.load("data/cln025.pdb")
    return (coords, forces, pdb, kbt)


def gen_config_map(pdb: md.Trajectory, string: str) -> LinearMap:
    """Create the configurational map.

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


def main() -> Tuple[TMap,...]:
    """Create Gaussian maps.

    This function
    """
    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map, we create a carbon alpha map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    #create a simple gaussian map
    results = project_forces(
        coords=coords,
        forces=forces,
        coord_map=cmap,
        constrained_inds=constraints,
        kbt=kbt,
        method=joptgauss_map, # specifies 
        l2_regularization=1e1,
        var=0.003, # sets variance of the added gaussian noise
    )
    # results is a dict with the derived map under "tmap"

    #create a staged gaussian map
    staged_results = project_forces(
        coords=coords,
        forces=forces,
        coord_map=cmap,
        constrained_inds=constraints,
        kbt=kbt,
        method=stagedjoptgauss_map, # specifies the staged map
        premap_l2_regularization=1e1, # regularization works differently
        l2_regularization=1e0,
        var=0.003,
    )
    # results is a dict with the derived map under "tmap"

    # this is the optimized trajectory map
    return (results["tmap"], staged_results["tmap"])


if __name__ == "__main__":
    """Create Gaussian map."""
    mapping = main()
    # we do nothing with the output
