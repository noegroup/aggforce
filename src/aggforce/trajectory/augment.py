"""Provides requirements for Augmenters.

Augmenters are used to extend the phase space of Trajectory objects.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from numpy import ndarray


class Augmenter(ABC):
    r"""Requirements for Augmenter classes.

    Given samples `x` from distribution `f`, Augmenter objects characterize
    distribution `g(x,y)` where `\int g(x,y) dy = f(x)`. They must be able to
    sample from the conditional distribution `(g(x,y)/f(x))` for a given `x`,
    and need to be able to evaluate the `\grad log g(x,y)/f(x)`, where `\grad`
    is taken with respect to both `x` and `y`. These tasks must be achieved
    without access to `f`.

    In practice, this means that if you give a Augmenter distribution a numpy
    array of points, it needs to be able to stochastically output new points
    conditioned on the input. For example, it may generate variates which
    are noised versions of the input. It must also

    Sample generation is handled by the `.sample` method and log gradient
    generation is handled by the `log_gradient` method.

    """

    @abstractmethod
    def __init__(self) -> None:
        r"""Initialize.

        Parameters controlling the behavior of the transform should be passed
        here.
        """

    @abstractmethod
    def sample(self, source: ndarray) -> ndarray:
        r"""Generate augmenting positions based on source positions.

        Arguments:
        ---------
        source:
            Source coordinates to use when sampling; i.e. the `x` in `(g(x,y)/f(x))`.
            Of shape (n_frames,n_particles,n_dims).

        Returns:
        -------
        Generated coordinates as np.ndarray; i.e. the `y` in `(g(x,y)/f(x))`. Of shape
        `(n_frames,n_particles,n_dims)`; `n_frames` and `n_dims` should be the same
        as source, but `n_particles` may differ. Shape should not change between calls.

        """

    @abstractmethod
    def log_gradient(
        self, source: ndarray, generated: ndarray
    ) -> Tuple[ndarray, ndarray]:
        r"""Evaluate log gradient of given augmentations and source coordinates.

        An Augmenter object models the conditional density `g(x,y)/f(x)`. This method
        evaluates `\grad_{x,y} log g(x,y)/f(x)` at a given `x,y`.

        Arguments:
        ---------
        source:
            See generated.
        generated:
            `source` corresponds to `x` in the description and `generated` corresponds
            to `y`. This function takes log derivatives at the point specified by these
            two arguments. They are split into two parts as `source` and `generated`
            are treated differently by other objects.

        Returns:
        -------
        Tuple, where first element is the portion of the log gradient taken with respect
        to `source` and the second is the log gradient with respect to `generated`. Both
        are taken at the position given by `source` and `generated` combined.

        """
