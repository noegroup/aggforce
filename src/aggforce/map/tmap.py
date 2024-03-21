r"""Provides maps for trajectory objects.

These objects effectively map coordinates and forces together.
"""

from typing import (
    Tuple,
    Callable,
    Final,
    Iterable,
    TypeVar,
    Optional,
    Any,
)
from abc import ABC, abstractmethod
from warnings import warn
import numpy as np
from ..trajectory import (
    CoordsTrajectory,
    ForcesTrajectory,
    Trajectory,
    AugmentedTrajectory,
    Augmenter,
)
from .core import CLAMap

ArrayTransform = Callable[[np.ndarray], np.ndarray]


_T_TMap = TypeVar("_T_TMap", bound="TMap")


class TMap(ABC):
    r"""Provides requirements to map Trajectory instances.

    TMaps differ from array maps in that they can easily express maps involving
    both coordinates and forces.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize."""

    @abstractmethod
    def __call__(self, t: Trajectory) -> Trajectory:
        """Map Trajectory to new instance."""

    def map_arrays(
        self, coords: np.ndarray, forces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map a coordinate and force arrays to their new version.

        This method wraps the __call__ method of the object.

        Arguments:
        ---------
        coords:
            Coordinate array. Should be of shape (n_frames,n_sites,n_dims).
        forces:
            Forces array. Should have the same shape as coords.

        Returns:
        -------
        2-tuple, where first element is the mapped coords and the second
        element is the mapped force array. Each is of the shape
        (n_frames,new_n_sites,n_dims).

        """
        t = Trajectory(coords=coords, forces=forces)
        derived = self(t)
        return (derived.coords, derived.forces)

    @abstractmethod
    def astype(self: _T_TMap, *args, **kwargs) -> _T_TMap:
        """Convert a TMap to a given type (numpy precision).

        The exact meaning of the type of a TMap depends on the implementation.
        However, if a TMap of type d is applied to input of type d, the
        output should have dtype d.

        Arguments are likely passed to numpy.astype calls.
        """


class SeperableTMap(TMap):
    r"""Creates a TMap from based on separate maps for both coordinates and forces."""

    def __init__(
        self,
        coord_map: ArrayTransform,
        force_map: ArrayTransform,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        coord_map:
            Callable that takes as input the coordinate array and returns the mapped
            coordinate array.
        force_map:
            Callable that takes as input the force array and returns the mapped
            coordinate array.

        """
        self.coord_map = coord_map
        self.force_map = force_map

    def __call__(self, t: Trajectory) -> Trajectory:
        """Map a trajectory using the saved maps.

        Arguments:
        ---------
        t:
            Trajectory to be mapped.

        Returns:
        -------
        Note that no matter what subclass of Trajectory is passed, a Trajectory
        is returned.

        """
        new_coords = self.coord_map(t.coords)
        new_forces = self.force_map(t.forces)
        return Trajectory(coords=new_coords, forces=new_forces)

    # we do not make this class generic, as unless it is overridden in a future
    # subclass, the returned type will not change.
    def astype(self, *args, **kwargs) -> "SeperableTMap":
        """Convert a SeperableTMap to a given type (numpy precision).

        This requires the underlying coord_map and force_map to themselves have
        a suitable astype method. If calling these underlying methods fails, a
        TypeError is raised.

        Arguments are passed to underlying tmaps via their astype method.
        """
        try:
            # we are catching the possibility of an attribute error.
            return self.__class__(
                coord_map=self.coord_map.astype(*args, **kwargs),  # type: ignore [attr-defined]
                force_map=self.force_map.astype(*args, **kwargs),  # type: ignore [attr-defined]
            )
        except AttributeError as e:
            raise TypeError(
                "Underlying coord_map and/or force_map do not support astype."
            ) from e


class CLAFTMap(TMap):
    """Trajectory map encompassing a force CLAMap and coordinate LinearMap.

    Used to create a trajectory map when the force map is configuration
    dependent.
    """

    def __init__(self, coord_map: ArrayTransform, force_map: CLAMap) -> None:
        """Initialize.

        Arguments:
        ---------
        coord_map:
            Callable that maps the coordinates using only the coords as input.
        force_map:
            CLAMap used to map the forces using the coords as copoints.

        """
        self.coord_map = coord_map
        self.force_map = force_map

    def __call__(self, t: Trajectory) -> Trajectory:
        """Map a Trajectory instance."""
        new_coords = self.coord_map(t.coords)
        new_forces = self.force_map(points=t.forces, copoints=t.coords)
        return Trajectory(coords=new_coords, forces=new_forces)

    # we do not make this class generic, as unless it is overridden in a future
    # subclass, the returned type will not change.
    def astype(self, *args, **kwargs) -> "CLAFTMap":
        """Convert a CLAFTMap to a given type (numpy precision).

        Arguments are passed to underlying tmaps via their astype method.

        This requires the underlying coord_map and force_map to themselves have
        a suitable astype method. If calling these underlying methods fails results
        in an AttributeError, a TypeError is raised.
        """
        try:
            # we are catching the possibility of an attribute error.
            return self.__class__(
                coord_map=self.coord_map.astype(*args, **kwargs),  # type: ignore [attr-defined]
                force_map=self.force_map.astype(*args, **kwargs),  # type: ignore [attr-defined]
            )
        except AttributeError as e:
            raise TypeError(
                "Underlying coord_map and/or force_map do not support astype."
            ) from e

        return self.__class__(forces=self.forces.astype(*args, **kwargs))


class AugmentedTMap(TMap):
    """Augments and then maps the resulting Trajectory.

    When mapping a trajectory, the input trjaectory is first transformed into a
    AugmentedTrajectory using augmenter and kbt; it is then mapped using a
    SeperableTMap map derived from aug_coord_map and aug_force_map.
    """

    def __init__(
        self,
        aug_tmap: TMap,
        augmenter: Augmenter,
        kbt: float,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        aug_tmap:
            Trajectory map that will be applied to each AugmentedTrajectory.
        augmenter:
            Expands trajectory prior to mapping, i.e. used to create each
            AugmentedTrajectory instance.
        kbt:
            Boltzmann constant multiplied by temperature for systems that will
            be mapped. Used to create each hAugmentedTrajectory instance.

        """
        self.tmap: Final = aug_tmap
        self.augmenter: Final = augmenter
        self.kbt: Final = kbt

    def __call__(self, t: Trajectory) -> Trajectory:
        """Map a Trajectory.

        The Trajectory is first transformed into an AugmentedTrajectory
        and then mapped.

        """
        augmented = AugmentedTrajectory.from_trajectory(
            t=t, kbt=self.kbt, augmenter=self.augmenter
        )
        return self.tmap(augmented)

    def astype(self, *args, **kwargs) -> "AugmentedTMap":
        """Convert a AugmentedTMap to a given type (numpy precision).

        Internal TMap and Augmenter instance are converted using their respective astype
        methods.
        """
        return self.__class__(
            aug_tmap=self.tmap.astype(*args, **kwargs),
            augmenter=self.augmenter.astype(*args, **kwargs),
            kbt=self.kbt,
        )


class ComposedTMap(TMap):
    """Combines multiple TMap instances into a single TMap.

    The given TMaps are applied one by one. Note that the rightmost
    map is applied first.

    Attributes:
    ----------
    submaps:
        List of the TMaps that are applied via a call method. Note
        that the maps are applied starting at the right of this list,
        not the left (this mirrors how functional composition is written.

        submaps may be modified after initialization.

    This object can be indexed;  integer indexing returns the underlying
    maps.

    """

    def __init__(
        self,
        submaps: Iterable[TMap],
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        submaps:
            Trajectory map that will be applied to each AugmentedTrajectory.

        """
        self.submaps: Final = list(submaps)

    def __call__(self, t: Trajectory) -> Trajectory:
        """Map a Trajectory.

        Each underlying TMap is applied (starting with the right-most map).

        """
        result = t
        for mapping in reversed(self.submaps):
            result = mapping(result)
            print(result.coords,result.forces)
        return result

    def __getitem__(self, idx: int, /) -> TMap:
        """Extract one of the underlying TMaps."""
        return self.submaps[idx]

    def astype(self, *args, **kwargs) -> "ComposedTMap":
        """Convert a ComposedTMap to a given type (numpy precision).

        Arguments are passed to each underlying tmap via its astype method.
        """
        new_maps = []
        for mapping in self.submaps:
            new_maps.append(mapping.astype(*args, **kwargs))
        return self.__class__(submaps=new_maps)


_T_Coords = TypeVar("_T_Coords", bound=CoordsTrajectory)


class NullForcesTMap(TMap):
    def __init__(self, warn_input_forces: bool = True, fill_value: Any = np.nan):
        self.warn_input_forces = warn_input_forces
        self.fill_value = fill_value

    def __call__(
        self,
        t: CoordsTrajectory,
    ) -> Trajectory:
        """Map Trajectory to new instance."""
        if isinstance(t, ForcesTrajectory):
            if self.warn_input_forces:
                warn("Discarding forces on input trajectory.", stacklevel=0)

        return Trajectory(coords=t.coords, forces=self.fill_value * t.coords)

    def map_arrays(
        self,
        coords: np.ndarray,
        forces: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map arrays using coord_map.

        This method mirrors

        Arguments:
        ---------
        coords:
        forces:

        Returns:
        -------
        mapped arrays

        """
        if forces is None:
            t = CoordsTrajectory(coords=coords)
        else:
            t = Trajectory(coords=coords, forces=forces)
        derived = self(t)
        return (derived.coords, derived.forces)

    def astype(self,*args,**kwargs) -> "NullForcesTMap":
        return self.__class__(warn_input_forces=self.warn_input_forces,
                              fill_value=self.fill_value)


class RATMap:
    """Maps the real portions of an AugmentedTrajectory.

    The augmented particles are _preserved_.
    """

    def __init__(self, tmap: TMap) -> None:
        """Initialize.

        Arguments:
        ---------
        tmap:
            TMap applying to the real particles.
        """
        self.tmap = tmap

    def __call__(self, t: AugmentedTrajectory) -> Trajectory:
        """Map a trajectory.

        Note that this function returns a Trajectory and not an AugmentedTrajectory.
        """
        # isolate read particles. This includes noise contributions!
        real_coord_entries = t.coords[:, t.real_slice, :]
        real_force_entries = t.forces[:, t.real_slice, :]
        # map real portions
        coords, forces = self.tmap.map_arrays(real_coord_entries, real_force_entries)
        # concatenate with noise particles
        full_coords = np.concatenate([coords, t.coords[:, t.aug_slice, :]], axis=1)
        full_forces = np.concatenate([forces, t.forces[:, t.aug_slice, :]], axis=1)
        return Trajectory(coords=full_coords, forces=full_forces)
