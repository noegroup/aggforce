"""Demonstrates cross validation for a linear optimized force map.

Example production script for running cross validation for a
configurationally dependent force map. Note that this may be computationally
expensive.

This example module does not produce optimized forces. Instead, it reports how
various hyperparameter choices affect the holdout force residual. Once optimal
hyperparameters are found, the user should then use those parameters for force
map generation.

In order to use this script, get_data should likely be modified to grab numpy
arrays of coordinates, forces, and pdbs from an appropriate filesystem
location. If not studying CLN025 at the carbon alpha resolution, gen_config_map
should also be modified.

Notes and documentation are provided throughout to give a better understanding of
the usage of various parts of the main code.

NOTE: You must have JAX installed to run this script.
"""

from typing import (
    Dict,
    Mapping,
    TypeVar,
    Hashable,
    Tuple,
    Any,
    List,
    Sequence,
    NamedTuple,
    Union,
)
import re
from pathlib import Path
from itertools import product
import numpy as np
import mdtraj as md  # type: ignore [import-untyped]
import pandas as pd  # type: ignore [import-untyped]
from copy import copy

# cross validation tools
from aggforce.agg import project_forces_grid_cv

# tools for preparing map optimization call
from aggforce import LinearMap, guess_pairwise_constraints

# tools for featurizing the system for nonlinear map creation
from aggforce.qp import (
    GeneralizedFeaturizer,
    Multifeaturize,
    qp_feat_linear_map,
    id_feat,
    gb_feat,
)

# helper for making featurizers
from aggforce.util import Curry

# number of folds of cross validation to do
NFOLDS = 5


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


# defines sane default parameters for the featurization function. These were
# found to produce okay results in previous runs on CLN025.
default_feat_args = {
    "inner": 0.0,
    "outer": 8.0,
    "width": 1.0,
    "n_basis": 8,
    "batch_size": 5000,
    "lazy": True,
}


def gen_feater(*args: Any, **kwargs: Any) -> GeneralizedFeaturizer:
    """Create a composite featurization function.

    Resulting featurizer has both identity features and configurationally dependent
    features.  These are implemented as two different functions, so we use provided
    tools to "glue" the two featurization functions together (Multifeaturize). We also
    associate arguments with the configurationally dependent (gb_feat) featurizer using
    Curry.

    Arguments:
    ---------
    *args:
        These are wrapped into the gb_feat featurizer
    **kwargs:
        These are wrapped into the gb_feat featurizer

    Returns:
    -------
    A combined featurizer (technically, a Multifeaturize object).
    """
    prod_kwargs = copy(default_feat_args)
    prod_kwargs.update(kwargs)
    # Curry takes a function and arguments, and returns a callable object that
    # stores the provided arguments.  That callable object, when called,
    # evaluates the original function using the stored arguments. We use it to
    # set default options for our our featurizer.
    f0 = Curry(gb_feat, *args, **prod_kwargs)
    # Multifeaturize takes a list of featurizers and returns a featurizer that
    # combines their output.
    # id_feat is the featurizer that produces a one-hot vector for each atom
    # that roughly encodes the index of that atom (constraints make it more
    # complicated).
    return Multifeaturize([id_feat, f0])  # type: ignore [list-item]


def gen_feater_grid(**kwargs: Any) -> List[GeneralizedFeaturizer]:
    """Create a list of featurization functions which have preset parameters.

    These parameters are chosen as all possible combinations of the arguments.
    This function calls gen_feater, and as a result produces featurizers that
    are a composite of id_feat and gb_feat: they have both one-hot id features
    and configurationally dependent features.

    Arguments:
    ---------
    **kwargs:
        each should be a list of possible values for a hyperparameter that has
        the same argument names. These values are combined to produce the
        featurization functions.

        For example: gen_feater_grid(a=[1,2,3],b=[4,5,6])
            produces featurizers that have the following kwargs values baked in:
                        a=1,b=4
                        a=1,b=5,
                        ...
                        a=3,b=5
                        a=3,b=6

    Returns:
    -------
    List of featurizers with values baked in via Curry.
    """
    arg_keys, arg_values = zip(*((x, y) for x, y in kwargs.items()))
    grid_iter = product(*arg_values)
    featers = []
    for values in grid_iter:
        kwargs_instance = dict(zip(arg_keys, values))
        featers.append(gen_feater(**kwargs_instance))
    return featers


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def merge(*dicts: Mapping[K, V]) -> Dict[K, V]:
    """Combine multiple dictionaries into a single dictionary.

    Arguments:
    ---------
    *dicts (Mappings):
        Dictionaries/Mappings to be combined. If keys are duplicated across
        dictionaries, the right most dictionary's element will be present in the
        output.

    Results:
    -------
    A single dictionary containing all the key-value pairs in the arguments
    """
    new: Dict[K, V] = {}
    for i in dicts:
        new.update(i)
    return new


def tabulate(dicts: Sequence[Mapping[K, V]]) -> pd.DataFrame:
    """Transform a iterable of mappings into a panda.

    Arguments:
    ---------
    dicts (Sequence (e.g., list) of dictionaries):
        Each dictionary should have the same keys, and is assumed to represent ta
        single row of the resulting DataFrame.

        Must have at least one element.

    Returns:
    -------
    pandas.DataFrame that has values from each dicts member in each row. Column
    names are the keys of the dictionaries.
    """
    keys = list(dicts[0].keys())
    content = {}
    for key in keys:
        to_add = []
        for d in dicts:
            to_add.append(d[key])
        content[key] = to_add
    return pd.DataFrame(content)


def make_df(cv_results: Dict[str, Dict[Any, Any]], key: str = "scores") -> pd.DataFrame:
    """Transform cross validation results into a readable DataFrame.

    This function is written based on what is known about the featurizations applied in
    this module. It does not apply to the generic output of the cross validation
    routines.

    Arguments:
    ---------
    cv_results (dictionary):
        The typical output of the *cv* routine called in this module.
    key (string):
        Passed to .sort_values() of the table we generate. Likely specifies a column to
        sort by.

    Returns:
    -------
    pandas DataFrame which has a column for each hyperparameter and a column for
    the value indexed under key. Usually, key is "scores" and this final value
    has the cross validation force residual. Featurization functions are treated
    specially and are transformed into a vector containing their baked in
    (Curried) arguments.
    """
    content = cv_results[key]
    rows = []
    for label, value in content.items():
        func_args = label.featurizer.featurizers[1].kwargs
        l2_reg = label.l2_regularization
        data_row = merge(func_args, {"l2": l2_reg, key: value})
        rows.append(data_row)
    tab = tabulate(rows)
    # we here set the index to be the hypers
    ind = set(tab.columns)
    ind.remove(key)
    tab.set_index(list(ind), inplace=True)
    tab.sort_values(key)
    return tab


def prune(tab: pd.DataFrame) -> pd.DataFrame:
    """Prune a DataFrame by removing redundant columns.

    Takes a pandas DataFrame and removes columns which only have a single
    unique value. A convenience function for make things more readable.
    """
    for col in tab.columns:
        if len(tab.loc[:, col].unique()) == 1:
            tab.drop(col, axis=1, inplace=True)
    return tab


def main() -> None:
    """Sample function for find a nonlinear force map via cross validation.

    Production script for finding a good configurationally
    dependent force map for CLN025 using cross validation. We do not use the
    derived parameters to actually map the forces.

    We perform the following big-picture steps here:
       - get data
       - make a configurational CG map
       - guess molecular constraints
       - generate an optimal force map that does not depend on configuration
         as a reference
       - generate a set of possible featurizers to use for a configurationally
         dependent force map
       - run cross validation over these possible featurizers coupled with
         possible l2_regularization values
       - transform the cross validation output into a more readable DataFrame
         and save it
    """
    coords, forces, pdb, kbt = get_data()
    # cmap is the configurational coarse-grained map
    cmap = gen_config_map(pdb, "CA$")
    # guess molecular constraints
    constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

    # before we run our cross validation over various possible hyperparameters
    # of the configurationally dependent map, we generate a optimized but static
    # force map for comparison. Ideally, we want to find force maps that are
    # significantly better than this static one. We could find such a map by
    # calling functions from linearmap, we also can use the featlinearmap calls
    # and only provide one-hot identity features as we do below. This is
    # almost equivalent, but l2_regularization is implemented in slightly
    # different ways in the two methods.

    # we do this using the cross validation function so that the residual it
    # reports is comparable to the residual we generate during our real cross
    # validation.

    # qp_feat_linear_map is the general method for configurationally dependent
    # maps and id_feat are the one-hot features (that make it effectively
    # configurationally INdepdendent here). The result is a map that has a
    # optimizable force coefficient for each atom. If this is confusing, just
    # note that we could instead call qp_linear_map and get effectively the same
    # reference value and force map.

    control_featted_results = project_forces_grid_cv(
        xyz=coords,
        forces=forces,
        config_mapping=cmap,
        constrained_inds=constraints,
        method=qp_feat_linear_map,
        # cv_arg_dict contains the parameters we will scan various arguments
        # over. Here, we just use a single l2 value, so it only "scans" over a
        # single value.
        cv_arg_dict={"l2_regularization": [1e3]},
        # passing in the one-hot features that don't depend on configuration
        featurizer=id_feat,
        kbt=kbt,
        n_folds=NFOLDS,
    )

    # now we move on to running cross validation over configurationally
    # dependent maps. We scan over the featurizer used and the l2_regularization
    # imposed. The featurizer is a single argument, so we first generate a
    # "grid" of featurizers, each of which has a different hyperparameter set.

    # more expression CV scanning set:
    # ruff doesn't like commented out code, but this is worth keeping.

    # featurizers =
    # gen_feater_grid(inner=[0.0, 1.0, 2.0, 3.0],
    #    outer=[7.0, 8.0, 9.0], # noqa: ERA001
    #    n_basis=[8, 9], # noqa: ERA001
    #    width=[0.7, 1.0, 1.3], # noqa: ERA001)
    # l2_regs = [5e1, 1e2] # noqa: ERA001

    featurizers = gen_feater_grid(
        inner=[0.0],
        outer=[7.0],
        n_basis=[5],
        width=[5],
    )
    l2_regs = [1e2, 1e3]
    # we then combine the l2_regs and featurizer lists into a single dictionary.
    # This is that is passed to the cross validation function, which will scan
    # over each combination from the two list entries
    cv_grid_basis: Dict[str, Union[List[GeneralizedFeaturizer], List[float]]] = {
        "featurizer": featurizers,
        "l2_regularization": l2_regs,
    }

    # this is the call that actually does the cross validation over the possible
    # configurationally dependent map optimizations.
    featted_results: Dict[str, Dict[NamedTuple, Any]] = project_forces_grid_cv(
        cv_arg_dict=cv_grid_basis,
        forces=forces,
        xyz=coords,
        n_folds=NFOLDS,
        config_mapping=cmap,
        constrained_inds=constraints,
        # again, we use our configurationally dependent method, but unlike
        # before our features do changes as a function of configuration, so our
        # force map will be configurationally dependent
        method=qp_feat_linear_map,
        # passing in the various featurization functions and l2_regs to scan
        # over
        kbt=kbt,
    )

    # summary is a more readable DataFrame of the results of the cross
    # validation
    key = "scores"
    summary = make_df(featted_results, key=key)
    reference_score = min(y for x, y in control_featted_results[key].items())
    summary["adjusted"] = summary[key] - reference_score
    summary.to_csv("cv.csv")
    prune(summary).to_csv("pruned_cv.csv")


if __name__ == "__main__":
    main()
