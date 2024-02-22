r"""Provides objects describing molecular maps.

These maps transform fine-grained points to coarse-grained points. Points may be
positions or forces.
"""

from abc import ABC, abstractmethod
from itertools import combinations, product
import numpy as np
from typing import Union, Dict, List, Iterable, Callable, Final


class Map(ABC):
    r"""Provides a abstract interface for maps.

    Maps transform fine-grained to coarse-grained systems.

    This is an abstract class. You must instantiate a subclass to use it. It
    mostly useful to remind what content can be assumed to be present for a
    Map object.

    Note that Map objects act on systems of particles, each of which has n
    dimensions (and not flat vectors).
    """

    n_dim: Final = 3

    @abstractmethod
    def __init__(self, tags: Union[None, Dict[str, str]]) -> None:
        """Initialize map.

        Arguments:
        ---------
        tags (dict):
            Used to create self.tags, which is a dictionary that than be
            arbitrarily populated at initialization. Useful for optimization
            methods to pass logs and residuals when a proper interface does not
            exist.
        """
        if tags is None:
            tags = {}
        self.tags = tags
        return

    @property
    @abstractmethod
    def n_cg_sites(self) -> int:
        r"""Number of coarse-grained sites in output of map."""

    @property
    @abstractmethod
    def n_fg_sites(self) -> int:
        r"""Number of fine-grained sites in input to map."""

    @abstractmethod
    def __call__(
        self, points: np.ndarray, copoints: Union[np.ndarray, None]
    ) -> np.ndarray:
        r"""Apply map to 3-dim array.

        Arguments:
        ---------
        points (np.ndarray):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).
        copoints (np.ndarray or None):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).
            Some mapping methods require an additional set of points. For
            example, positions must sometimes be given in order to apply a
            position-dependent force map.

        Returns:
        -------
        Mapped version of points as a three dimensional nd.numpy array.
        """


class LinearMap(Map):
    r"""Unified interface for linear maps.

    Allows multiple different representations of the same map to be used.
    Various representations are converted into a consistent internal
    representation, which is then transformed into other formats when needed.

    The primary map format is the "standard_matrix". Given that our linear map
    transforms configurations from the fine-grained (fg) resolution to the
    coarse-grained (cg) resolution, the standard matrix is a (num. of cg
    particles) x (num. of fg particles) where each element describes how a fg
    particle linearly contributes to a cg particle. This can be accessed using
    the standard_matrix attribute.

    Calling instances allows them to map trajectory arrays of the shape
    (n_steps,n_sites,n_dims).
    """

    def __init__(
        self,
        mapping: Union[List[List[int]], np.ndarray],
        n_fg_sites: Union[int, None] = None,
        tags: Union[None, Dict[str, str]] = None,
    ) -> None:
        r"""Initialize LinearMapping from something describing a map.

        Arguments:
        ---------
        mapping (list of lists of integers or 2-d numpy.ndarray):
            If a list of lists, then we assume the outer list iterates over
            various cg indices, and the inner lists describe the indices of
            which atoms contribute to that particular cg site. As this format
            does not make it clear how many total fg sites there are, n_fg_sites
            must be specified. If instead a 2-d numpy.ndarray, then the array is
            assumed to be of shape (num of cg,num of fg), which each element
            describing the coefficient of how the fg site contributes to the cg
            site. In this case n_fg_sites should not be specified.
        n_fg_sites (integer or None):
            Certain mapping descriptions make it ambiguous how many total
            fine-grained sites there are. This variable allows this ambiguity to
            be resolved.
        tags (dictionary or None):
            Passed to Map init.

        Example:
        -------
            [[0,2,3],[4]] with n_fg_sites=6 describes a 6 particle fg system
            and 2 particle cg system (from the length of the outer list).
            cg particle 0 (equally) depends on fg particles 0,2, and 3 as where
            cg particle 1 depends only on fg particle 4.

            The same information is given by the following 2-d matrix:
                [ 1/3 0   1/3 1/3 0   0  ]
                [ 0   0   0   0   1   0  ]
            Note that in the matrix case, we had to specify the normalization of
            the weights directly, as where in the list format it was done
            automatically.
        """
        super().__init__(tags)

        if isinstance(mapping, np.ndarray) and len(mapping.shape) == 2:
            if n_fg_sites is not None:
                raise ValueError()
            self._standard_matrix = mapping
        elif hasattr(mapping, "__iter__"):
            # assume we are in the case of iterable of lists
            if n_fg_sites is None:
                raise ValueError()
            mapping = list(mapping)
            n_cg_sites = len(mapping)
            mapping_mat = np.zeros((n_cg_sites, n_fg_sites))
            for site, site_contents in enumerate(mapping):
                local_map = np.zeros(n_fg_sites)
                local_map[site_contents] = 1 / len(site_contents)
                mapping_mat[site, :] = local_map
            self._standard_matrix = mapping_mat
        else:
            raise ValueError()

    @property
    def standard_matrix(self) -> np.ndarray:
        r"""The mapping in standard matrix format."""
        return self._standard_matrix

    @property
    def n_cg_sites(self) -> int:
        r"""Number of coarse-grained sites in output of map."""
        return self._standard_matrix.shape[0]

    @property
    def n_fg_sites(self) -> int:
        r"""Number of fine-grained sites in input of map."""
        return self._standard_matrix.shape[1]

    @property
    def participating_fg(self) -> List[List[int]]:
        r"""Table of atoms included in cg site.

        Formatted as a dictionary of {cg_index:<list_of_fg_indices>}.
        """
        inc_matrix = self.standard_matrix > 0
        ind_form = list(zip(*np.nonzero(inc_matrix)))
        table: List[List[int]] = []
        for _ in range(self.n_cg_sites):
            table.append([])
        for cg_ind, fg_ind in ind_form:
            table[cg_ind].append(fg_ind)
        return table

    def __call__(
        self,
        points: np.ndarray,
        copoints: Union[np.ndarray, None] = None,  # noqa: ARG002
    ) -> np.ndarray:
        r"""Apply map to a particular form of 3-dim array.

        Arguments:
        ---------
        points (np.ndarray):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).
        copoints:
            Ignored. Included for compatibility with parent class.

        Returns:
        -------
        Combines points along the n_sites dimension according to the internal
        map.
        """
        return trjdot(points, self.standard_matrix)


class CLAMap(Map):
    r"""Provide representation of a Co-Local Affine map.

    An affine transformation on x is Ax+b (a translation and linear
    transformation). Consider two time series, x_i and y_i. Consider mapping x_i
    using information from y_i to create a specific affine transformation for
    every x_i as so:
        Input:
            x_1, x_2, ..., x_T
            (with y_1, y_2, ..., y_T)
        Output:
            A(y_1)x_1+b(y1), A(y_2)x_2+b(y_2), ..., A(y_T)x_T+b(y_T)

    This is the type of map this object describes. It is specified by providing
    A(.) and b(.). A is referred to as scale, and b is referred to as trans.  We
    refer to the values being mapped as points, and the values creating A and b
    copoints (hence the name co-local affine map).

    This is a particular form of nonlinear map and is the output of featurized
    force maps. Unfortunately, it cannot be expressed using a single matrix
    (unlike linear maps), and so does not have a standard_matrix property.
    """

    def __init__(
        self,
        scale: Callable,
        trans: Callable,
        n_fg_sites: int,
        n_cg_sites: Union[None, int] = None,
        zeroes_check: bool = True,
        tags: Union[Dict, None] = None,
    ) -> None:
        r"""Initialize CLAMap object from functions scale (A) and trans (b).

        Arguments:
        ---------
        scale (callable):
            Callable which accepts an array of shape (n_steps,n_fg_sites,n_dim)
            and returns (n_steps,n_fg_sites,n_cg_sites). See class description for
            more details.
        trans (callable):
            Callable which accepts an array of shape (n_steps,n_fg_sites,n_dim)
            and returns (n_steps,n_cg_sites,n_dim). See class description for
            more details.
        n_fg_sites (integer):
            Number of fg sites in the input space.
        n_cg_sites (integer or None):
            Number of cg sites in the output space. If set to None, then
            zero_check must be True, and it's value is inferred during the zero
            zero check.
        zeroes_check (boolean):
            If true, A and b are tested by creating a zeros array for both the
            points and copoints. The dimension of these arrays corresponds to a
            single frame trajectory and uses the n_fg_sites argument and n_dim
            class variable.
        tags (dictionary or None):
            Passed to Map init.
        """
        super().__init__(tags)

        if zeroes_check:
            z_points = np.zeros((1, n_fg_sites, Map.n_dim))
            mapped = trjdot(z_points, scale(z_points)) + trans(z_points)
            if n_cg_sites is None:
                n_cg_sites = mapped.shape[1]
            else:
                if n_cg_sites != mapped.shape[1]:
                    raise ValueError("n_cg_sites did not match results from zero test")
        else:
            if n_cg_sites is None:
                raise ValueError(
                    "If n_cg_sites is not set, zeroes_check must be truthy."
                )

        self._n_cg_sites = n_cg_sites
        self._n_fg_sites = n_fg_sites
        self.scale = scale
        self.trans = trans

    @property
    def n_cg_sites(self) -> int:
        r"""Number of coarse-grained sites in map output."""
        return self._n_cg_sites

    @property
    def n_fg_sites(self) -> int:
        r"""Number of fine-grained sites in map input."""
        return self._n_fg_sites

    def __call__(
        self, points: np.ndarray, copoints: Union[None, np.ndarray]
    ) -> np.ndarray:
        r"""Apply map to 3-dim array.

        Arguments:
        ---------
        points (np.ndarray):
            Assumed to be 3 dimensional of shape (n_steps,n_sites,n_dims).
        copoints (np.ndarray):
            Provided as input to A and b during mapping. Assume dot be of form
            (n_steps,n_sites,n_dims). See class description.

        Returns:
        -------
        Combines points along the n_sites dimension according to the internal
        map functions.
        """
        scale = self.scale(copoints)
        trans = self.trans(copoints)
        return trjdot(points, scale) + trans


def trjdot(points: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Perform a matrix product with mdtraj-style arrays.

    Arguments:
    ---------
    points (numpy.ndarray):
        3-dim ndarray of shape (n_steps,n_sites,n_dims). To be mapped using
        factor.
    factor (numpy.ndarray):
        2-dim ndarray of shape (n_cg_sites,n_sites) or 3-dim ndarray of shape
        (n_steps,n_cg_sites,n_sites). Used to map points.

    Returns:
    -------
    ndarray of shape (n_steps,n_cg_sites,n_dims) contained points mapped
    with factor.

    Notes:
    -----
    Functionality is most easily described via an example:
        Molecular positions (and forces) are often represented as arrays of
        shape (n_steps,n_sites,n_dims). Other places in the code we often
        transform these arrays to a reduced (coarse-grained) resolution where
        the output is (n_steps,n_cg_sites,n_dims).

        (When linear) the relationship between the old (n_sites) and new
        (n_cg_sites) resolution can be described as a matrix of size
        (n_sites,n_cg_sites). This relationship is between sites, and is
        broadcast across the other dimensions. Here, the sites are contained in
        points, and the mapping relationship is in factor.

        However, we cannot directly use dot products to apply such a matrix map.
        This function applies this factor matrix as expected, in spirit of
        (points * factor).

        Additionally, if instead the matrix mapping changes at each frame of the
        trajectory, this can be specified by providing a factor of shape
        (n_steps,n_cg_sites,n_sites). This situation is determined by
        considering the dimension of factor.
    """
    # optimal path found from external optimization with einsum_path
    opt_path = ["einsum_path", (0, 1)]
    if len(factor.shape) == 2:
        return np.einsum("tfd,cf->tcd", points, factor, optimize=opt_path)
    if len(factor.shape) == 3:
        return np.einsum("...fd,...cf->...cd", points, factor, optimize=opt_path)
    raise ValueError("Factor matrix is an incompatible shape.")


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
