r"""Routines for for featurized non-linear force map optimization.

NOTE: Maps of this type are more complex than simple configurationally-independent force
maps and have shown any benefit. For typical force mapping, stick to linear maps.

Types and routines for general feature calculation are included.

Configurationally dependent (non-linear) force maps may be made by using a featurization
function which characterizes each configuration. The force map may then be made linear
to this featurization. The linear coefficients do not change as a function of
configuration, but the output of the featurizers do.

This module provides routines for optimizing this kind of force map. Other modules
provide additional featurization functions.
"""
from typing import (
    Any,
    Union,
    Tuple,
    List,
    Callable,
    Final,
    ClassVar,
    Generator,
    Iterable,
    TypeVar,
    overload,
    Literal,
)
from typing_extensions import TypedDict
from queue import SimpleQueue, Empty
from copy import deepcopy
import numpy as np
from numpy.random import default_rng
import scipy.sparse as ss  # type: ignore [import-untyped]
from qpsolvers import solve_qp  # type: ignore [import-untyped]
from ..map import CLAFTMap, LinearMap, CLAMap
from ..constraints import Constraints, reduce_constraint_sets
from ..trajectory import Trajectory
from .qplinear import SolverOptions, DEFAULT_SOLVER_OPTIONS

# This defines featurizers and their output via types.

# featurizers output a dictionary with these keys.
KNAME_FEATS: Final = "feats"
KNAME_DIVS: Final = "divs"
KNAME_NAMES: Final = "names"
# we also have to spell these literals out below because of mypy limits.
Features = TypedDict(
    "Features",
    {
        "feats": Iterable[np.ndarray],
        "divs": Iterable[np.ndarray],
        "names": Union[Iterable[str], None],
    },
)

# here we characterize the featurizers themselves. GeneralizedFeaturizes include
# complex lists of generators which act like Features. GeneralizedFeaturizers output
# those or features.
Featurizer = Callable[[np.ndarray, LinearMap, Constraints], Features]
GeneralizedFeatures = Union[Features, "FeatZipper"]
# first option here actually does need a union in its output because of mypy
GeneralizedFeaturizer = Union[
    Callable[[np.ndarray, LinearMap, Constraints], Union[Features, "FeatZipper"]],
    Featurizer,
]

T = TypeVar("T")
CallableT = TypeVar("CallableT", bound=Callable)


class FeatZipper:
    r"""Lazily combines the output of multiple featurizers.

    NOTE: This function does not combine _featurizers_; it combines their
    output. To combine featurizers use multifeaturize.

    Featurizers output a dict with keys for features ("feats"), divergences
    ("divs"), and names ("names"). This class combines the _output_ of multiple
    featurizers, providing dictionary-like interface for accessing features and
    divergences aggregated across all the results. This class can be used with
    lazy featurizers (those providing generators) and is itself lazy: "feats" and
    "divs" index generators.

    In other words, instances of this class are objects which can be indexed
    using the same keys as featurizer output. Indexing returns generators that
    iterate over the data aggregated from the featurized content provided at
    initialization.

    The information for the produced feature and divergence generators comes
    from shared data present in each instance. The original data are iterated
    over as needed (in the case of lazy features, this implies that memory and
    computation is performed only when needed by the aggregate output) and
    served through the provided generators.
    """

    # these keys provide understanding into the feature dictionary format
    generator_keys = frozenset([KNAME_FEATS, KNAME_DIVS])
    name_key = "names"

    # joiners has the particular functions for combining arrays from each
    # content member's iteration. They are indexed by the appropriate key.
    # e.g., "feats"'s callable combines iterations from "feats" entries.

    # members cannot be static methods and in this dictionary, so they are
    # lambdas.
    joiners: ClassVar = {
        KNAME_FEATS: lambda args: np.concatenate(args, axis=2),
        KNAME_DIVS: lambda args: np.concatenate(args, axis=1),
    }

    def __init__(self, content: List[GeneralizedFeatures]) -> None:
        r"""Initialize a FeatZipper from list of content dictionaries.

        Arguments:
        ---------
        content (list of dictionaries):
            List of the dictionaries that  we want to aggregate and iterate
            over. Each dictionary should be the output of a featurizer; that is,
            it should have "feats", "divs", and "names" as keys.  "feats" and
            "divs" should index an iterable of numpy arrays and "names" should
            be a list of strings.
        """
        self.reset(content)
        # no current support for names
        self.names = None

    def keys(self) -> frozenset:
        r"""Return a frozenset of all viable keys for indexing.

        Returns:
        -------
        Frozenset of all feasible keys.
        """
        return self.generator_keys.union(frozenset([KNAME_NAMES]))

    def reset(self, content: Iterable[GeneralizedFeatures]) -> None:
        r"""Prepare internal state.

        Internal state includes setting up iterators and queues. The queues are
        used to temporarily aggregate the output of content's iterators when
        necessary. The queues are used to aggregate results as we (indirectly)
        iterate over the input iterables in content.

        Arguments:
        ---------
        content (list of dicts):
            list of dictionaries, each of which is the output of a
            featurization function.  See __init__'s content argument for more
            details.
        """
        self.source = {}
        for key in self.generator_keys:
            # mypy cannot check to see if we are obeying the named dictionary
            self.source.update({key: zip(*[o[key] for o in content])})  # type: ignore
        queues: List[SimpleQueue] = [SimpleQueue() for _ in self.generator_keys]
        self._queues = dict(zip(self.generator_keys, queues))

    def _makegenerator(self, key: str) -> Generator[Any, None, None]:
        r"""Create generator for a specified key.

        These generators query the internal queues for results from the input
        dictionaries. If the queues are empty, they try to repopulate them; if
        that fails, they return.

        Arguments:
        ---------
        key (str):
            The key used to extract the series from the content dictionaries for
            aggregation.

        Yields:
        ------
        Aggregated numpy.ndarrays formed from results under key in content dictionaries.

        Examples:
        --------
        If "feats" is passed, we return a generator; at each iteration
        this generator grabs an iteration from the iterator under "feats" from
        each content dictionary, combines them into a single array, and returns them.
        """
        while True:
            try:
                item = self._queues[key].get(block=False)
            except Empty:
                try:
                    self._populate(key=key, exception=False)
                    item = self._queues[key].get(block=False)
                except Empty:
                    return
            yield item

    def __getitem__(self, key: str) -> Union[Generator[Any, None, None], None]:
        r"""Return generator associated with key or (in the case of names) None.

        Arguments:
        ---------
        key (str):
            A member of self.generator_keys or NAME_KEY. If
            NAME_KEY, then self.names is returned. If a member of
            generator_keys, a generator which iterates over the aggregated
            form of that key's content in self.content. See the class
            description for more details.

        Returns:
        -------
        Generator or self.names
        """
        if key in self.generator_keys:
            return self._makegenerator(key)
        if key == KNAME_NAMES:
            return self.names
        raise KeyError("Invalid key; valid keys are {}".format(self.keys()))

    def _populate(self, key: Union[str, None] = None, exception: bool = True) -> None:
        r"""Add item to queues by continuing source generators one step.

        The internal queues' purposes are to store the output of the input
        feature data for processing. This method is called when other methods
        realize the queues are empty.

        Arguments:
        ---------
        key (str or None):
            Specifies which queue to refill. Should be one of the keys in
            self.generator_keys. If None, all queues refilled.
        exception (boolean):
            If true, StopIteration exceptions are not caught; as a result, if
            the internal iterators are depleted StopIteration will be thrown. If
            false, this situation is caught and the method simply leaves the
            queues untouched and returns.
        """
        if key is None:
            keys = self.generator_keys
        else:
            keys = frozenset([key])
        for key in keys:
            try:
                data = next(self.source[key])
                agg = self.joiners[key](data)
                self._queues[key].put(agg)
            except StopIteration as e:
                if exception:
                    raise e
        return


def qp_feat_linear_map(
    traj: Trajectory,
    coord_map: LinearMap,
    featurizer: Featurizer,
    kbt: float,
    n_constraint_frames: int = 20,
    constraints: Union[None, Constraints] = None,
    sparse: bool = True,
    solver_args: SolverOptions = DEFAULT_SOLVER_OPTIONS,
    l2_regularization: float = 1e1,
) -> CLAFTMap:
    r"""Search for force map with lowest mean square norm of the mapped force.

    The produced map is linear with respect to user-provided features.

    Arguments:
    ---------
    traj (Trajectory):
        Trajectory instance to extract coordinates and forces from.
    coord_map (map.LinearMap):
        LinearMap object which characterizes configurational map.
    featurizer (callable):
        A callable for featurizing the system which uses the following as input:
            copoints (numpy.ndarray):
                3-D of shape (n_frames,n_fg_sites,n_dims=3) containing the
                copoints (positions) used for featurization
            coord_map (map.LinearMap instance)
                configurational coarse-graining map of the system
            constraints (set of frozensets)
                Characterizes molecular constraints in the system
        and return a dictionary with the following key-value pairs:
            'feats': list/generator  of feature mats (n_frames, n_fg_sites, n_feats)
                (see below for more information)
            'divs': list/generator of divergence mats (n_frames, n_feats, n_dim)
                (see below for more information)
            'names': None or list of name strings (n_feat)
        where n_feat is the dimensionality of the features. Note that this must
        perform all of the featurizations for all sites simultaneously.
    kbt (float):
        Boltzmann constant times the temperature of the system. The units must
        match the energy portion of the forces units. Needed to account for the
        divergence correction.
    n_constraint_frames (positive integer):
        Number of frames to sample when constraining featurization index to obey
        orthogonality constraints between coarse-grained sites. A larger number
        provides a more exact constraint, but increases memory usage.
    constraints (set of frozensets of integers):
        Each entry is a frozenset of indices, the group of which is constrained.
        Currently, only bond constraints (frozensets of size 2) are supported.
    sparse (boolean):
        If true, then various matrices generated in the method are sparsified
        before being passed to the solver. This does not typically reduce the
        runtime memory usage but may remove warnings given by solvers.
    solver_args (dict):
        Passed as options to qp_solve to solve quadratic program via **.
    l2_regularization (float):
        Coefficient for penalizing the l2 norm of the parameter vector during
        quadratic programming. Note that this l2 penalty is not the same as
        that present in the simple linear map optimization: here, we penalize
        the size of the coefficient vector, as where there we penalize the
        resulting mapping. Doing so here is more complex as the mapping changes
        as a function of frame.

    Notes:
    -----
    Uses a quadratic programming solver with equality constraints.

    Additionally, the featurization function output must conform to the
    following format. The entry for the 'feat' and 'div' keys is a list, and
    each element in each list corresponds to the features that are used for a
    particular coarse-grained site; e.g., the first entry in both the feat and
    div entries correspond to coarse-grained site index 0.

    Each feats list entry is of the shape (n_frames, n_fg_sites, n_feats), where
    n_frames is the number of trajectory frames, n_fg_sites is the number of
    fine-grained sites in the system, and n_feats is the number of features
    produced (each fine-grained site in each frame has multiple features).

    Each div list entry characterizes divergence and has the shape (n_frames,
    n_feats, n_dim). In this the feature for every fg_site and frame is d'd
    w.r.t each Cartesian component (this means 3 times as many values as the
    features, but is then summed over the fine-grained sites).

    See the documentation for exp2_feat for more information.

    Returns:
    -------
    CLAFTMap instance characterizing the combination of the given coordinate
    and derived force mapping.

    """
    if constraints is None:
        constraints = set()

    feat_results = featurizer(traj.coords, coord_map, constraints)
    feats, divs, names = [
        feat_results[key] for key in [KNAME_FEATS, KNAME_DIVS, KNAME_NAMES]  # type: ignore
    ]

    per_site_feat_coef = []
    for ind, (feat, div) in enumerate(zip(feats, divs)):
        # make constraint arrays
        constr_mult, constr_target = _constr_arrays(
            features=feat,
            cg_ind=ind,
            coord_map=coord_map,
            n_frames=n_constraint_frames,
            sparse=sparse,
        )

        # create the force aggregation part of the matrix
        # <forces> * <dim mat> * <features> matrix
        force_features = np.einsum("...af,...ad->...fd", traj.forces, feat)

        # <divergence> is already ready
        # mat1 + kbt mat 2
        # div is an inconvenient shape in the proofs
        reshaped_div = np.swapaxes(div, 1, 2)
        # combine agg'd forces and div term
        ms_reg_mat = force_features + kbt * reshaped_div
        reg_mat = np.reshape(ms_reg_mat, newshape=(-1, ms_reg_mat.shape[2]))
        qp_mat = np.matmul(reg_mat.T, reg_mat)
        if l2_regularization > 0:
            qp_mat += np.diag((l2_regularization,) * qp_mat.shape[0])
        if sparse:
            qp_mat = ss.csc_matrix(qp_mat)
        params = solve_qp(
            P=qp_mat,
            q=np.zeros(qp_mat.shape[0]),
            A=constr_mult,
            b=constr_target,
            **solver_args,
        )
        if params is None:
            raise ValueError("Map optimization failed.")
        per_site_feat_coef.append(params)

    force_map = _feat_linear_mapping(
        featurizer=featurizer,
        coefs=per_site_feat_coef,
        mapping=coord_map,
        constraints=constraints,
        tags={"feat_names": names, "coef_list": per_site_feat_coef},
    )

    return CLAFTMap(coord_map=coord_map, force_map=force_map)


def _constr_arrays(
    features: np.ndarray,
    cg_ind: int,
    coord_map: LinearMap,
    n_frames: int,
    sparse: bool = True,
) -> Tuple[Union[np.ndarray, ss.csc_matrix], np.ndarray]:
    """Produce a constraint array and target array for constrained optimization.

    Produce a constraint array (A) and target 1-array (b) for later Ax=b
    constrained quadratic optimization.

    The returned 2-array is of shape (n_cg_sites*n_frames,n_features) and the
    1-array is of shape (n_cg_sites*n_frames,). Is sparse is true, the first
    array is a sparse csc matrix.

    Arguments:
    ---------
    features (numpy.ndarray):
        Array characterizing the features for a single coarse-grained site.
        Assumed to be of shape (n_frames,n_fg_sites,n_features).
    cg_ind (positive integer):
        Later optimization is performed for each cg site individually. The
        particular cg site index used there is also passed here, and is used
        when creating the output target 1-array.
    coord_map (map.LinearMap instance):
        Configurational mapping characterizing the coarse-grained resolution.
        Used when creating the output 2-array (including setting n_cg_sites).
    n_frames (positive integer):
        Number of frames to randomly sample from features when constructing
        the constraint matrix. Larger values use more memory, but provide a more
        exact constraint in later optimization.
    sparse (boolean):
        If true, the 2-array returned is a sparse csc array. This may
        avoid warnings from various external numerical solvers. Note that
        algebraic operations on this output may change as it is a matrix and not
        and array.

    Returns:
    -------
    Ordered pair: first element is a (n_cg_sites*n_frames,n_features) sized 2-array
    (or scipy.sparse.csc_matrix, see below), second element is 1-dimensional array
    of shape (n_cg_sites*n_frames,).

    If sparse, a sparse csc_matrix is returned for the first array. Note that
    the algebraic operations change as it is a matrix and not and array.
    """
    # get random subset of calculated features
    frame_indices = default_rng().choice(len(features), size=n_frames, replace=False)
    subsetted_features = features[frame_indices]

    mult_array = np.einsum(
        "ca,...af->...cf", coord_map.standard_matrix, subsetted_features
    )
    target_array = np.zeros((n_frames, coord_map.n_cg_sites))
    target_array[:, cg_ind] = 1

    cmshape = mult_array.shape
    mult = np.reshape(mult_array, (-1,) + cmshape[-1:])
    target = np.reshape(target_array, (-1,))
    if sparse:
        mult = ss.csc_matrix(mult)
    return (mult, target)


def _feat_linear_mapping(
    featurizer: Featurizer,
    coefs: List[np.ndarray],
    mapping: LinearMap,
    constraints: Constraints,
    **kwargs
) -> CLAMap:
    """Create CLAMap for a linearly parameterized map.

    Creates a CLAMap instance for a map that is constructed from a fixed
    parameterization and linear coefficients.

    Arguments:
    ---------
    featurizer (callable):
        A callable for featurizing the system which uses the following as input:
            copoints (numpy.ndarray):
                3-D of shape (n_frames,n_fg_sites,n_dims=3) containing the
                copoints (positions) used for featurization
            coord_map (map.LinearMap instance)
                configurational coarse-graining map of the system
            constraints (set of frozensets)
                Characterizes molecular constraints in the system
        and return a dictionary with the following key-value pairs:
            'feats': list of feature mats (n_frames, n_fg_sites, n_feats)
                (see below for more information)
            'divs': list of divergence mats (n_frames, n_feats, n_dim)
                (see below for more information)
            'names': None or list of name strings (n_feat)
        where n_feat is the dimensionality of the features. Note that this must
        perform all of the featurizations for all sites simultaneously.
    coefs (list of 1-D numpy arrays):
        List of 1-D numpy arrays which are combined with the outputted features
        in the spirit of matmul(<feat>,<coefs>).  Note that these do not _need_
        to be the same sized arrays, but rather must be compatible with the
        output of featurizer (which may vary in dimensionality across
        coarse-grained sites).
    mapping (LinearMap instance):
        Passed to featurizer and used to extract number of coarse-grained sites.
    constraints (set of frozensets):
        Each entry is a frozenset of coarse-grained indices, the group of which
        is constrained. Passed to featurizer.
    **kwargs:
        Passed to CLAMap initialization.

    Returns:
    -------
    CLAMap instance.
    """

    def scale_f(copoints: np.ndarray) -> np.ndarray:
        feats = featurizer(copoints, mapping, constraints)["feats"]
        weights = [np.einsum("...ij,j->...i", f, c) for f, c in zip(feats, coefs)]
        return np.stack(weights, axis=1)

    def trans_f(copoints: np.ndarray) -> np.ndarray:
        divs = featurizer(copoints, mapping, constraints)["divs"]
        weights = [np.einsum("tij,i->tj", f, c) for f, c in zip(divs, coefs)]
        return np.stack(weights, axis=1)

    force_mapping = CLAMap(
        scale=scale_f,
        trans=trans_f,
        n_fg_sites=mapping.n_fg_sites,
        zeroes_check=True,
        **kwargs,
    )

    return force_mapping


@overload
def id_feat(
    points: np.ndarray,
    cmap: LinearMap,
    constraints: Constraints,
    return_ids: Literal[False],
) -> GeneralizedFeatures:
    ...


@overload
def id_feat(
    points: np.ndarray,
    cmap: LinearMap,
    constraints: Constraints,
    return_ids: Literal[True],
) -> np.ndarray:
    ...


def id_feat(
    points: np.ndarray,
    cmap: LinearMap,
    constraints: Constraints,
    return_ids: bool = False,
) -> Union[np.ndarray, GeneralizedFeatures]:
    """Feature a one-hotted label to each fine-grained site.

    Vector valued feature which associates a one-hotted label to each
    fine-grained site.  Labels are unique, except that they are shared between
    constraint groups.

    Arguments:
    ---------
    points (numpy.ndarray):
        Points characterizing the positions of fine-grained sites as a function
        of time; assumed to be of shape (n_frames,n_sites,n_dims=3). Note that
        this is only used to find the number of frames of the system, but is
        included for compatibility with other featurization functions.
    cmap (map.LinearMap):
        Configurational map characterizing the relationship between the
        coarse-grained and fine-grained resolutions.
    constraints (set of frozensets of integers):
        Each frozenset describes fine-grained sites participating in a
        constraint. These constraint subsets may overlap.
    return_ids (boolean):
        If return_ids is True, return a numpy.ndarray containing the constraint
        d/group of each fine-grained site. Useful for creating other features
        which respect constraints, but not a feature in itself.

    Returns:
    -------
    If return_ids is True, return a numpy.ndarray with the shape (n_fg_sites,).
    Else: A dict with the following key-value pairs:
        'feats': list of feature mats (n_frames, n_fg_sites, total_n_feats)
            The features do not changed between frames.
        'divs': list of div mats (n_frames, total_n_feats, n_dim).
            These are filled with zeros as the features do not change as a
            function of position.
        'names': None

    NOTE: While the features are not generated lazily (they are not returned via
    generators), the entries for each cg_site are views to the same numpy array,
    so the memory footprint is only that of a single cg_site.
    """
    # get list of groups of fine-grained sites which have to share id features
    # because of constraints
    groups = deepcopy(constraints)
    groups = groups.union(frozenset([x]) for x in range(cmap.n_fg_sites))
    reduced_groups = sorted(reduce_constraint_sets(groups))
    places = []
    if return_ids:
        ids = np.zeros(cmap.n_fg_sites, dtype=np.int32)
        for label, fg_set in enumerate(reduced_groups):
            indices = list(fg_set)
            ids[indices] = label
        return ids
    else:
        for label, fg_set in enumerate(reduced_groups):
            for fg_ind in fg_set:
                places.append([fg_ind, label])

    n_frames = points.shape[0]
    n_fg_sites = cmap.n_fg_sites
    n_cg_sites = cmap.n_cg_sites
    n_types = len(reduced_groups)
    n_dim = cmap.n_dim

    inds = list(zip(*places))
    feats = np.zeros((n_frames, n_fg_sites, n_types), dtype=np.float32)
    feats[:, inds[0], inds[1]] = 1

    divs = np.zeros((n_frames, n_types, n_dim), dtype=np.float32)

    return {"feats": [feats] * n_cg_sites, "divs": [divs] * n_cg_sites, "names": None}


def multifeaturize(featurizers: List[GeneralizedFeaturizer]) -> GeneralizedFeaturizer:
    """Combine multiple featurization functions into a single featurization function.

    Arguments:
    ---------
    featurizers (list of callables):
        A list, each member of which must use the following as callable input:
            copoints (numpy.ndarray):
                3-D of shape (n_frames,n_fg_sites,n_dims=3) containing the
                points used for featurization
            coord_map (map.LinearMap instance)
                configurational coarse-graining map of the system
            constraints (set of frozensets)
                Characterizes molecular constraints in the system
        and return a dictionary with the following key-value pairs:
            'feats': list of feature mats (n_frames, n_fg_sites, n_feats)
            'divs': list of divergence mats (n_frames, n_feats, n_dim)
            'names': None or list of name strings (n_feat)

    Returns:
    -------
    Callable which returns the output expected of a featurization function: a
    dictionary with the following key value pairs:
       'feats': list of feature mats (n_frames, n_fg_sites, total_n_feats)
       'divs': list of divergence mats (n_frames, total_n_feats, n_dim)
       'names': None
    This output is created by combining the pertinent arrays along the n_feat
    dimensions.  Input is the same as it was for each individual featurization
    function.

    Note:
    ----
    Output names are not supported and are set to None.
    """

    def composite(
        copoints: np.ndarray, coord_map: LinearMap, constraints: Constraints
    ) -> GeneralizedFeatures:
        output = [feat(copoints, coord_map, constraints) for feat in featurizers]
        return FeatZipper(content=output)

    return composite


class Multifeaturize:
    """Lazily combine featurizers into a single featurizer.

    Combine multiple featurizers into a single featurizer by creating a
    derivative callable object. Combination is done lazily where possible.

    Usage:
        m = Multifeaturize([feat1,feat2])
        combined_features = m(args)

        where combined_features are featurization results that aggregate over
        the output of feat1 and feat2 and args are the same args that would have
        been passed to feat1/feat2. Note that the combined_features is lazy (it
        uses generators).

    This is an object and not a closure to allow for self description.
    """

    def __init__(self, featurizers: Iterable[GeneralizedFeaturizer]) -> None:
        """Initialize Multifeaturize object from a list of featurizers.

        Arguments:
        ---------
        featurizers (list of callables):
            List of featurization callables. See qp_feat_linear_map for more
            information. Featurizers have the following type signature:


        Returns:
        -------
        Callable which evaluates and aggregates the output from each element of
        featurizers.
        """
        self.featurizers = featurizers

    def __str__(self) -> str:
        """Generate string representation."""
        sp = "    "
        msg = []
        msg.append("{} instance:".format(self.__class__))
        for ind, func in enumerate(self.featurizers):
            msg.append("Callable {}:".format(ind))
            func_msg = [sp + o for o in str(func).split("\n")]
            msg.extend(func_msg)
        return "\n".join(msg)

    def __repr__(self) -> str:
        """Generate brief string representation."""
        msg = []
        msg.append("{}():".format(self.__class__))
        for ind, func in enumerate(self.featurizers):
            msg.append("C{}:".format(ind))
            func_msg = repr(func)
            msg.append(func_msg)
        return " ".join(msg)

    def __call__(self, *args, **kwargs) -> GeneralizedFeatures:
        """Evaluate each featurizer and return the wrapped output.

        Arguments:
        ---------
        *args:
            Passed to each featurizer.
        **kwargs:
            Passed to each featurizer.

        Returns:
        -------
        FeatZipper object wrapping the output of all featurizers.
        """
        results = [f(*args, **kwargs) for f in self.featurizers]
        return FeatZipper(content=results)
