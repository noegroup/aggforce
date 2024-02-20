r"""Provides interface for optimally aggregating forces.

Methods are described in the following problem setting:

We have a fine grained system (with n_fg particles) which we map to
coarse-grained system (with n_cg particles) using a linear mapping function. The
configurational portion of this map is already set; methods here provide ways to
calculate the force map.

project_forces creates a suitable force map using a variety of methods (linear and
nonlinear). project_forces_grid_cv performs cross validation.
"""

from typing import (
    Union,
    Set,
    FrozenSet,
    Callable,
    Dict,
    Any,
    Collection,
    List,
    NamedTuple,
    Tuple,
    Mapping,
    TypeVar,
    Final,
)
from gc import collect
from itertools import product
import numpy as np
from . import linearmap
from . import constfinder
from .map import LinearMap


PROJECT_FORCES_CNSTR_AUTO: Final = "auto"

SCORES_KNAME: Final = "scores"
SDS_KNAME: Final = "sds"
NRUNS_KNAME: Final = "n_runs"

PROJFORCE_KNAME: Final = "projected_forces"
FMAP_KNAME: Final = "map"
RESIDUAL_KNAME: Final = "residual"
CONSTRAINTS_KNAME: Final = "constraints"


def project_forces(
    xyz: Union[None, np.ndarray],
    forces: np.ndarray,
    config_mapping: LinearMap,
    constrained_inds: Union[Set[FrozenSet[int]], str, None] = PROJECT_FORCES_CNSTR_AUTO,
    method: Callable[..., Callable] = linearmap.qp_linear_map,
    **kwargs,
) -> Dict[str, Any]:
    r"""Produce optimized force map.

    NOTE: Performs convenience operations (e.g., making sure the mapping matrix
    is in the correct form) so that internal methods can have strong assumptions
    about arguments.

    Arguments:
    ---------
    xyz (np.ndarray):
        Three dimensional array of shape (n_steps,n_sites,n_dims). Contains the
        positions of the fg sites as a function of time.  Note that in the case
        of linear force maps, the content of this argument is ignored for
        finding forces. However, if constrained_inds is set to 'auto', it may
        still be used to find possible constraints.
    forces (np.ndarray):
        Three dimensional array of shape (n_steps,n_sites,n_dims). Contains the
        forces on the fg sites as a function of time.
    config_mapping (map.LinearMap):
        LinearMap characterizing the fg -> cg configurational map.
    constrained_inds (set of frozensets or 'auto'):
        If a set of frozensets, then each entry is a frozenset of indices, the
        group of which is constrained.  Currently, only bond constraints (frozen
        sets of 2 elements) are supported.  if 'auto', then
        guess_pairwise_constraints is used to generate a list of constrained
        atoms. All of xyz is passed to this function; if more flexibility is
        desired, call it externally and pass its output through this argument.
    method (callable):
        Specifies what method to use to find the optimal map.
    kwargs:
        Passed to method.

    Returns:
    -------
    If only_return_forces, return an np.ndarray of shape
    (n_steps,n_cg_sites,n_dims) which contains the optimally mapped forces.
    If not only_return_forces, a dictionary with the following elements is
    returned:
        projected_force =
            np.ndarray of shape (n_steps,n_cg_sites,n_dims).
        map =
            Map characterizing the optimal force map.
        residual =
            Force map residual calculated using force_smoothness. Note that this
            is not performed on a hold-out set, so be wary of overfitting.
        constraints =
            Set of frozensets characterizing the molecular constraints on the
            system. Useful if constrained_inds is set to 'auto'.

    Notes:
    -----
    The strings used as keys in the output dictionary are source from the following
    submodule variables:
        PROJFORCE_KNAME
        MAP_KNAME
        RESIDUAL_KNAME
        CONSTRAINTS_KNAME

    """
    if constrained_inds == PROJECT_FORCES_CNSTR_AUTO:
        if isinstance(xyz, np.ndarray):
            constrained_inds = constfinder.guess_pairwise_constraints(xyz)
        else:
            raise ValueError(
                f"If constrained_inds is {PROJECT_FORCES_CNSTR_AUTO}, "
                "xyz cannot be None."
            )
    force_map: Callable = method(
        xyz=xyz,
        config_mapping=config_mapping,
        forces=forces,
        constraints=constrained_inds,
        **kwargs,
    )
    mapped_forces = force_map(points=forces, copoints=xyz)
    to_return = {}
    to_return.update({PROJFORCE_KNAME: mapped_forces})
    to_return.update({FMAP_KNAME: force_map})
    to_return.update({RESIDUAL_KNAME: force_smoothness(mapped_forces)})
    to_return.update({CONSTRAINTS_KNAME: constrained_inds})
    return to_return


T = TypeVar("T")


def project_forces_grid_cv(
    cv_arg_dict: Mapping[str, List[T]],
    forces: np.ndarray,
    xyz: Union[np.ndarray, None] = None,
    n_folds: int = 5,
    **kwargs,
) -> Dict[str, Dict[NamedTuple, T]]:
    """Cross validation over project_forces using a grid of parameters.

    Note: this function does not choose an optimal model. Instead, it performs
    cross validation for each parameter listed in cv_arg_dict. You should use
    this to select an optimal hyperparameter and then train a production model.

    Arguments:
    ---------
    cv_arg_dict (dictionary):
        Contains arguments to run cross validation over. Must be of the
        following (limited) form: {<argument_name>:[arg_val_1,arg_val2,...]}.
        Each val is passed to project_forces as argument_name=arg_val_1.
    forces (numpy.ndarray):
        See project_forces; it is split into CV folds before being passed to
        project_forces.
    xyz (numpy.ndarray or None):
        See project_forces; it is split into CV folds before being passed to
        project_forces, unless it is None, in which case it is simply passed.
    n_folds (positive integer):
        Number of cross validation folds to use.
    *args:
        Passed to project_forces.
    **kwargs:
        Passed to project_forces.

    Returns:
    -------
    dictionary composed of a series of dictionaries containing
    <parameters>:<holdout score> pairs, where parameter is each is
    force_smoothness evaluated each fold and then averaged. 'scores' indexes the
    for mean force fluctuation values, 'sds' indexes their sample standard
    deviations, and 'n_runs' indexes the number of optimization runs that
    completed successfully. If no runs successfully finish, then the standard
    deviation and mean entries are set to None. <parameters> is represented by a
    custom NamedTuple-derived instance.
    """
    # make fold indices
    n_frames = forces.shape[0]
    frames = np.arange(n_frames)
    np.random.default_rng().shuffle(frames)
    chunked_frame_inds = np.array_split(ary=frames, indices_or_sections=n_folds, axis=0)

    # create sequence of indices which are outside each fold (for training)
    compl_chunked_frame_inds = []
    for ind, _ in enumerate(chunked_frame_inds):
        outside_chunks = [x for i, x in enumerate(chunked_frame_inds) if i != ind]
        compl_chunked_frame_inds.append(np.concatenate(outside_chunks))

    procced_cv_args = process_cvargs(cv_arg_dict)
    cv_results: Dict[str, Dict[Any, Any]] = {
        SCORES_KNAME: {},
        SDS_KNAME: {},
        NRUNS_KNAME: {},
    }
    # iterate over values of parameter
    for cv_arg_label, cv_arg_dict in procced_cv_args:
        cv_fold_scores = []
        combined_kwargs = dict(kwargs, **cv_arg_dict)
        # iterate over folds
        for train_inds, val_inds in zip(compl_chunked_frame_inds, chunked_frame_inds):
            # make training data
            train_forces = forces[train_inds]
            if xyz is None:
                train_xyz = None
            else:
                train_xyz = xyz[train_inds]
            # use training data for parameterization
            try:
                trained_map = project_forces(
                    xyz=train_xyz, forces=train_forces, **combined_kwargs
                )[FMAP_KNAME]
                # make validation data
                val_forces = forces[val_inds]
                if xyz is None:
                    val_xyz = None
                else:
                    val_xyz = xyz[val_inds]
                # use validation data
                cv_fold_scores.append(
                    force_smoothness(trained_map(points=val_forces, copoints=val_xyz))
                )
                del trained_map
            except ValueError as e:
                print(e)
            collect()
        cv_results[SCORES_KNAME].update({cv_arg_label: mean(cv_fold_scores)})
        cv_results[SDS_KNAME].update({cv_arg_label: sample_sd(cv_fold_scores)})
        cv_results[NRUNS_KNAME].update({cv_arg_label: len(cv_fold_scores)})
    return cv_results


def process_cvargs(
    arg_dict: Mapping[str, List[Any]]
) -> List[Tuple[NamedTuple, Dict[str, Any]]]:
    """Transform argument values into a "grid" of parameter combinations.

    Arguments:
    ---------
    arg_dict (dictionary):
        Arguments to process. Assumed to have the form
        {
            key1: [key1_arg1,key1_arg2,...]
            key2: [key1_arg1,key1_arg2,...]
            ...
        }
        where key* are the names of the arguments, and key*_arg* are the
        argument values.

    Returns:
    -------
    A list, where entries are tuples of the form (using the example above):
        (<namedtuple>(key1=key1_arg1, key2=key1_arg1),
                                    {key1:key1_arg1, key2:key2_arg_1})
        (<namedtuple>(key1=key1_arg2, key2=key2_arg1) :
                                    {key1:key1_arg1, key2:key1_arg_1})
                ...
    i.e., the first element of each tuple is a namedtable derived instance and
    the second element is a dictionary that can be passed containing command
    parametres.  An entry is given for every combination of parameters. The
    namedtuple entries are instances of a named tuple constructed to have a
    field for each parameter.
    """
    # NOTE: mypy has issues here. Type checking may be incorrect, and
    # certain types have been manually annotated with warnings ignored.

    # the parameter names we are going to make a grid over
    param_names = list(arg_dict.keys())
    # values the parameters can take
    values = [content for _, content in arg_dict.items()]
    cross_values = product(*values)
    to_return = []
    # mypy doesn't like dynamic named tuples like this
    CVArgs = NamedTuple("CVArgs", param_names)  # type: ignore [misc]
    for v in cross_values:
        # mypy also has a bug for this named tuple usage
        key = CVArgs(**dict(zip(param_names, v)))
        sub_args = {}
        for name in param_names:
            sub_args.update({name: getattr(key, name)})
        to_return.append((key, sub_args))
    # mypy is also confused here
    return to_return  # type: ignore [return-value]


def force_smoothness(array: np.ndarray) -> float:
    r"""Calculate mean squared element of an array.

    This is proportional to a finite sum approximate of E[||x||^2_2], which
    is often used as a metric of quality for force-maps.
    """
    return np.mean(array**2)


def mean(s: Collection[float]) -> Union[float, None]:
    """Compute arithmetic mean.

    Arguments:
    ---------
    s:
        Collection of floats to calculate the mean of.

    Returns:
    -------
    If s is empty, returns None; otherwise, returns mean.

    Notes:
    -----
    This is needed as a function for type checking.
    """
    if len(s) == 0:
        return None
    return sum(s) / len(s)


def sample_sd(s: Collection[float]) -> Union[float, None]:
    """Compute sample standard deviation.

    Arguments:
    ---------
    s:
        Collection of floats to calculate the sample standard deviation of.

    Returns:
    -------
    If s is empty, returns None; otherwise, returns mean.

    Notes:
    -----
    This is needed as a function for type checking.
    """
    m = mean(s)
    if m is None:
        return None
    sd = sum([(o - m) ** 2 for o in s])
    sd /= len(s) - 1
    sd = sd ** (0.5)
    return sd
