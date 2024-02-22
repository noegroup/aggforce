"""Provides basic jax tools used in other submodules.

This module should not have dependencies on other package submodules, and should
not pull in optional dependencies (e.g., jax).
"""
from typing import Union, Callable, TypeVar, Iterable, Any, List, Generic
import numpy as np

T = TypeVar("T")


def distances(
    xyz: np.ndarray,
    cross_xyz: Union[None, np.ndarray] = None,
    return_matrix: bool = True,
    return_displacements: bool = False,
) -> np.ndarray:
    """Calculate the distances for each frame in a trajectory.

    Returns an array where each slice is the distance matrix of a single frame
    of an argument.

    Arguments:
    ---------
    xyz (np.ndarray):
        An array describing the cartesian coordinates of a system over time;
        assumed to be of shape (n_steps,n_sites,n_dim).
    cross_xyz (np.ndarray or None):
        An array describing the Cartesian coordinates of a different system over
        time or None; assumed to be of shape (n_steps,other_n_sites,n_dim). If
        present, then the returned distances are those between xyz and cross_xyz
        at each frame.  If present, return_matrix must be truthy.
    return_matrix (boolean):
        If true, then complete (symmetric) distance matrices are returned; if
        false, the upper half of each distance matrix is extracted, flattened,
        and then returned.
    return_displacements (boolean):
        If true, then instead of a distance array, an array of displacements is
        returned.

    Returns:
    -------
    Returns numpy.ndarrays, where the number of dimensions and size depend on
    the arguments.

    If return_displacements is False:
        If return_matrix and cross_xyz is None, returns a 3-dim numpy.ndarray of
        shape (n_steps,n_sites,n_sites), where the first index is the time step
        index and the second two are site indices. If return_matrix and
        cross_xyz is not None, then an array of shape
        (n_steps,other_n_sites,n_sites) is returned. If not return_matrix,
        return a 2-dim array (n_steps,n_distances), where n_distances indexes
        unique distances.
    else:
        return_matrix must be true, and a 4 dimensional array is returned,
        similar to the shapes above but with an additional terminal axis for
        dimension.
    """
    if cross_xyz is not None and not return_matrix:
        raise ValueError("Cross distances only supported when return_matrix is truthy.")
    if return_displacements and not return_matrix:
        raise ValueError("Displacements only supported when return_matrix is truthy.")

    if cross_xyz is None:
        displacement_matrix = xyz[:, None, :, :] - xyz[:, :, None, :]
    else:
        displacement_matrix = xyz[:, None, :, :] - cross_xyz[:, :, None, :]
    if return_displacements:
        return displacement_matrix
    distance_matrix = np.linalg.norm(displacement_matrix, axis=-1)
    if return_matrix:
        return distance_matrix
    n_sites = distance_matrix.shape[-1]
    indices0, indices1 = np.triu_indices(n_sites, k=1)
    subsetted_distances = distance_matrix[:, indices0, indices1]
    return subsetted_distances


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


def flatten(nested_list: Iterable[Iterable[Any]]) -> List[Any]:
    """Flattens a nested list.

    Arguments:
    ---------
    nested_list (list of lists):
        List of the form [[a,b...],[h,g,...],...]

    Returns:
    -------
    Returns a list where the items are the subitems of nested_list. For example,
    [[1,2],[3,4] would be transformed into [1,2,3,4].
    """
    return [item for sublist in nested_list for item in sublist]


# this should be replaced by the call from functools
# type hinting may be improvable, but not clear because of old mypy.
def curry(func: Callable[..., T], *args, **kwargs) -> Callable[..., T]:
    """Curry a function using named and keyword arguments.

    For f(x,y), curry(f,y=a) returns a function g, where g(b) = f(x=b,y=a).
    Non-keyword arguments also work--- they are passed after any non-keyword
    arguments passed to g.  Useful when creating a featurization function with
    certain options set.

    Type hints only maintain return type.

    Arguments:
    ---------
    func (callable):
        Function to be curried.
    *args:
        Used to curry func.
    **kwargs:
        Used to curry func.

    Returns:
    -------
    Callable which evaluates func appending args and kwargs to any passed
    arguments.
    """

    def curried_f(*sub_args, **sub_kwargs) -> T:
        return func(*sub_args, *args, **sub_kwargs, **kwargs)

    return curried_f


# mypy limitations in the current version block paramspec usage
R = TypeVar("R")


class Curry(Generic[R]):
    """Callable class to curry a function.

    Uses named and keyword arguments.

    That is: for f(x,y), curry(f,y=a) returns a callable g, where g(b) =
    f(x=b,y=a). Non-keyword arguments also work--- they are passed after any
    non-keyword arguments passed to g.  Useful when creating a featurization
    function with certain options set.

    This is an object and not a closure to allow for self description.

    Type hints only maintain return value.
    """

    def __init__(self, func: Callable[..., R], *args, **kwargs) -> None:
        """Initialize a Curry object from a function and arguments.

        Arguments:
        ---------
        func (callable):
            Function to be curried.
        *args:
            Used to curry func.
        **kwargs:
            Used to curry func.

        Returns:
        -------
        Callable which evaluates func appending args and kwargs to any passed
        arguments.
        """
        self.args = args
        self.kwargs = kwargs
        self.func = func

    def __str__(self) -> str:
        """Generate string representation."""
        sp = "    "
        func_msg = [sp + o for o in str(self.func).split("\n")]
        args_msg = [sp + o for o in str(self.args).split("\n")]
        kwargs_msg = [sp + o for o in str(self.kwargs).split("\n")]
        msg = []
        msg.append("{} instance:".format(self.__class__))
        msg.append("callable:")
        msg.extend(func_msg)
        msg.append("args:")
        msg.extend(args_msg)
        msg.append("kwargs:")
        msg.extend(kwargs_msg)
        return "\n".join(msg)

    def __repr__(self) -> str:
        """Generate brief string representation."""
        func_msg = repr(self.func)
        args_msg = repr(self.args)
        kwargs_msg = repr(self.kwargs)
        msg = []
        msg.append("{}():".format(self.__class__))
        msg.append("C:")
        msg.append(func_msg)
        if len(self.args):
            msg.append("Ar:")
            msg.append(args_msg)
        if len(self.kwargs):
            msg.append("Kw:")
            msg.append(kwargs_msg)
        return " ".join(msg)

    def __call__(self, *sub_args, **sub_kwargs) -> R:
        """Call curried function."""
        return self.func(*sub_args, *self.args, **sub_kwargs, **self.kwargs)
