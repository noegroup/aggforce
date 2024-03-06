r"""Provides maps for trajectory objects.

These objects effectively map coordinates and forces together.
"""

from typing import Tuple, Callable, Final, Iterable
from abc import ABC, abstractmethod
import numpy as np
from ..trajectory import Trajectory, AugmentedTrajectory, Augmenter
from .core import CLAMap

ArrayTransform = Callable[[np.ndarray], np.ndarray]


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
        return result

    def __getitem__(self, idx: int, /) -> TMap:
        """Extract one of the underlying TMaps."""
        return self.submaps[idx]


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
