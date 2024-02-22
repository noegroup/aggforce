r"""Provides routines for the optimal linear force maps."""

from typing import Union
from typing_extensions import TypedDict
import numpy as np
from qpsolvers import solve_qp  # type: ignore [import-untyped]
from ..map import LinearMap
from ..constraints import Constraints, reduce_constraint_sets, constraint_lookup_dict

SolverOptions = TypedDict(
    "SolverOptions",
    {
        "solver": str,
        "eps_abs": float,
        "max_iter": int,
        "polish": bool,
        "polish_refine_iter": int,
    },
)
DEFAULT_SOLVER_OPTIONS: SolverOptions = {
    "solver": "osqp",
    "eps_abs": 1e-7,
    "max_iter": int(1e3),
    "polish": True,
    "polish_refine_iter": 10,
}


def qp_linear_map(
    forces: np.ndarray,
    config_mapping: LinearMap,
    constraints: Union[None, Constraints] = None,
    l2_regularization: float = 0.0,
    xyz: Union[np.ndarray, None] = None,  # noqa: ARG001
    solver_args: SolverOptions = DEFAULT_SOLVER_OPTIONS,
) -> LinearMap:
    r"""Search for optimal linear force map.

    Optimally is determined via  average lowest mean square norm of the mapped force.

    Note: Uses a quadratic programming solver with equality constraints.

    Arguments:
    ---------
    forces (np.ndarray):
        three dimensional array of shape (n_steps,n_sites,n_dims). Contains the
        forces of the FG sites as a function of time.
    config_mapping (np.ndarray):
        LinearMap object which characterizes configurational map.
    constraints (set of frozensets):
        Each entry is a frozenset of indices, the group of which is constrained.
        Currently, only bond constraints (frozensets of size 2) are supported.
    l2_regularization (float):
        if positive, a l2 normalization of the (full) mapping vector is applied
        with this coefficient.
    xyz (None):
        Ignored. Included for compatibility with the interface of other methods.
    solver_args (dict):
        Passed as options to qp_solve to solve quadratic program.

    Returns:
    -------
    LinearMap object characterizing force mapping.
    """
    if constraints is None:
        constraints = set()
    # flatten force array
    reshaped_fs = qp_form(forces)
    # construct geom constraint matrix
    # prep matrices for solver
    con_mat = make_bond_constraint_matrix(config_mapping.n_fg_sites, constraints)
    reg_mat = np.matmul(reshaped_fs, con_mat)
    qp_mat = np.matmul(reg_mat.T, reg_mat)
    zero_q = np.zeros(qp_mat.shape[0])
    per_site_maps = []
    # since we want to penalize the norm of the expanded vector, we add
    # con_mat.t*con_mat
    if l2_regularization > 0.0:
        qp_mat += l2_regularization * np.matmul(con_mat.T, con_mat)
    # run solver
    for ind in range(config_mapping.n_cg_sites):
        sbasis = np.zeros(config_mapping.n_cg_sites)
        sbasis[ind] = 1
        constraint_mat = np.matmul(config_mapping.standard_matrix, con_mat)
        gen_map = solve_qp(
            P=qp_mat, q=zero_q, A=constraint_mat, b=sbasis, **solver_args
        )
        per_site_maps.append(np.matmul(con_mat, gen_map))
    return LinearMap(np.stack(per_site_maps))


def qp_form(target: np.ndarray) -> np.ndarray:
    r"""Transform 3-array to a particular form of 2-array.

    e.g. target is (n_steps,n_sites,n_dims=3)
    output is (n_steps*n_dims,n_sites) where the rows are ordered as
        step=0, dim=0
        step=0, dim=1
        step=0, dim=2
        step=1, dim=0
    """
    mixed = np.swapaxes(target, 1, 2)
    reshaped = np.reshape(mixed, (mixed.shape[0] * mixed.shape[1], -1))
    return reshaped


def make_bond_constraint_matrix(n_sites: int, constraints: Constraints) -> np.ndarray:
    r"""Make constraint matrix connecting a generalized maps to the expanded maps.

    This matrix connects a generalized mapping coefficient to the expanded
    mapping coefficient.

    When creating optimal force maps, atoms which are molecularly constrained to
    each other are most easily handled by having them share mapping
    coefficients. We do so by creating a reduced mapping vector, that when
    multiplied by the matrix this function produces, creates a full sized
    mapping vector.  Optimization may then be performed over the reduced (or
    generalized) vector.

    This is done by creating a matrix that duplicates pertinent indices in the
    reduced vector over multiple atoms. This results in (as an example) a
    matrix like this:
        [1 0 0 0]
        [0 1 0 0]
        [0 1 0 0]
        [0 0 1 0]
        [0 0 0 1]
    when multiplied by a reduced vector [a b c d] on the right, this results in
    the vector [a b b c d], i.e., sites 1 and 2 are constrained to have the
    same value.

    Arguments:
    ---------
    n_sites (integer):
        Total number of sites in the system. In the context of the quadratic
        programming in this module, usually the number of fine-grained
        particles in the system.
    constraints (set of frozensets of integers):
        Each member set contains fine-grained site indices of atoms which are
        constrained relative to each other (and should have identical mapping
        coefficients)

    Returns:
    -------
    2-dim numpy.ndarray
    """
    # aggregate various constraints if needed
    rconstraints = reduce_constraint_sets(constraints)
    # get number of DOFs we will remove
    n_constrained_atoms = sum((len(x) for x in rconstraints))
    reduced_n_sites = n_sites - n_constrained_atoms + len(rconstraints)
    # make look up dictionary so that we know which site are dependent on which
    # atoms
    index_lookup = constraint_lookup_dict(rconstraints)
    mat = np.zeros((n_sites, reduced_n_sites))
    offset = 0
    # place all sites that don't depend on another site
    for site in range(n_sites):
        if site not in index_lookup:
            mat[site, offset] = 1
            offset += 1
    # place all sites that do depend on another site
    for site, anchor in index_lookup.items():
        mat[site, :] = mat[anchor, :]
    return mat
