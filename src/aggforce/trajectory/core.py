"""Provides tools and definitions for Trajectory objects."""

from typing import Tuple, Optional, Callable, overload, Literal, TypeVar, Any, NoReturn
from copy import deepcopy
from numpy import ndarray, concatenate
from .augment import Augmenter


class ForcesOnlyTrajectory:
    r"""Trajectory with forces but without positions.

    This is similar to Trajectory, but without forces. See Trajectory class for
    more information.
    """

    def __init__(self, forces: ndarray) -> None:
        """Initialize.

        Arguments:
        ---------
        forces:
            forces for multiple timesteps.
        """
        if len(forces.shape) != 3:
            raise ValueError("forces must have 3 dimensions.")
        self.forces = forces
        return

    @property
    def n_sites(self) -> int:
        """Number of particles in the system."""
        return self.forces.shape[1]

    @property
    def n_dim(self) -> int:
        """Dimension of the individual particles in the system.

        This is 3 in typical molecular dynamics applications.
        """
        return self.forces.shape[2]

    def __len__(self) -> int:
        """Return the number of frames in the system."""
        return len(self.forces)

    def __getitem__(self, index: slice) -> "ForcesOnlyTrajectory":
        """Index trajectory.

        Only slices are allowed. Returns a ForcesOnlyTrajectory instance.
        """
        if not isinstance(index, slice):
            raise ValueError("Only slices are allowed for indexing.")
        new_forces = self.forces[index]
        return self.__class__(forces=new_forces)

    def copy(self) -> "ForcesOnlyTrajectory":
        """Copy a trajectory object."""
        new_forces = self.forces.copy()
        return self.__class__(forces=new_forces)


class Trajectory(ForcesOnlyTrajectory):
    r"""Collection of coordinates and forces from a molecular trajectory.

    A molecular dynamics simulation saves coordinates and forces at various
    snapshots in time. This class encapsulates coordinates and forces for a
    sequence of snapshots. Minimal functionality over the arrays is
    implemented; this class has value as it allows for more complex Trajectory
    objects to be created.

    Attributes/Methods:
    ------------------
    coords:
        array of coordinates. Should have shape `(n_frames, n_sites, n_dim)`,
        where `n_frames` is the number of timesteps/snapshots in the data,
        `n_sites` is the number of particles or atoms in the system, and `n_dim`
        is the physical dimension the particles reside in (almost always `3`).
    forces:
        Array containing force information. Should be the same shape as `coords`.
        Should have compatible units with `coords`, although this is not needed
        in this object.
    n_sites:
        Property giving the number of sites in the trajectory.
    n_dim:
        Property giving the dimension the trajectory sites reside in.
    __len__:
        Number of snapshots or frames in the trajectory.
    __get_item__:
        Supports slicing, but not integer indexing.
    copy:
        Allows for copying the underlying arrays.

    """

    def __init__(self, coords: ndarray, forces: ndarray) -> None:
        """Initialize.

        Arguments:
        ---------
        coords:
            positions for multiple timesteps. Of shape (n_frames,n_sites,n_dims).
        forces:
            forces for multiple timesteps. Must match shape of coords.
        """
        if coords.shape != forces.shape:
            raise ValueError("coords and forces must be of same shape.")
        if len(coords.shape) != 3:
            raise ValueError("coords and forces must be of same shape.")
        self.coords = coords
        super().__init__(forces=forces)
        return

    def __getitem__(self, index: slice) -> "Trajectory":
        """Index trajectory.

        Only slices are allowed. Returns a Trajectory instance.
        """
        if not isinstance(index, slice):
            raise ValueError("Only slices are allowed for indexing.")
        new_coords = self.coords[index]
        new_forces = self.forces[index]
        return Trajectory(coords=new_coords, forces=new_forces)

    def copy(self) -> "Trajectory":
        """Copy a trajectory object."""
        new_coords = self.coords.copy()
        new_forces = self.forces.copy()
        return Trajectory(coords=new_coords, forces=new_forces)


A = TypeVar("A")


class AugmentedTrajectory(Trajectory):
    r"""Trajectory where part of the state space is generated via a given algorithm.

    If we take an existing trajectory dataset of coordinates and forces, we can
    add noise to the positions (more generally, stochastically augment them) to extend
    the phase space. This procedure allows us to create additional force information.

    Assuming that the original trajectory can be characterized by stationary density
    `f(x)`, we extend this trajectory by adding `y` to the state space. The resulting
    space has entries `(x,y)`, and is characterized via density `g(y|x)f(x)`, where
    we've added conditional density `g`. The "force" over this extended ensemble is
    proportional to gradient of the log of the extended density:
    ```
        \grad log [ g(y|x)f(x) ]
         = \grad log [ g(y|x) ] + \grad log [ f(x) ]
         = \grad log [ g(y|x) ] + \grad log [ exp( -[1/kbt] U(x) )]
         = \grad log [ g(y|x) ] + [1/kbt] \grad [ -U(x) ]
    ```
    Note that `kbt` is the Boltzmann constant times the temperature of the
    original sample and `U` is the energy function of the real system (assuming
    a NVT ensemble).  In order to remove the proportionality and obtain the
    forces of the extended ensemble we multiply all entries by `kbt`:
    ```
         \rightarrow kbt * \grad log [ g(y|x) ] + \grad [ -U(x) ]
    ```
    It is important to note that the forces on on the noise variables (`y`) have no
    contribution from `U`; however, the forces on the real positions (`x`) do:
    ```
        augmented_y_forces = kbt * \grad_y log [ g(y|x) ]
        augmented_x_forces = kbt * \grad_x log [ g(y|x) ] + \grad_x [ -U(x) ]
    ```

    This trajectory allows one to create trajectories over `(x,y)` that are
    compatible with non-augmented Trajectory instances.  This is performed by starting
    from a dataset composed of `(x, - \grad_x U(x))` pairs: these coincide with the
    coordinates (`x`) and forces (`- \grad_x U(x)`) associated with typical Trajectory
    objects.

    The augmenter entries below provide the log gradients of `g` and the
    ability to sample `y` from `g(.|x)`, as where `x` and `- \grad_x U(x)` are
    provided via the `coords` and `forces` arguments.

    Attributes/Properties:
    ---------------------
    augmenter:
        Augmenter instance that adds new random force and coordinate information to
        the system.
    coords:
        Augmented coordinates of the system `(x,y)`. Created using a single random
        draw from augmenter.
    forces:
        Augmented forces of the system `(x,y)`. Created using a single random
        draw from augmenter.
    kbt:
        Boltzmann constant multiplied by the temperature of the real system. This
        is used to scale the log gradients given by augmenter to have the same
        units as the entries in real_forces.
    refresh:
        Refreshes coords and forces via new augmenter calls.
    real_coords:
        The coordinates of the non-augmented particles.
    real_forces:
        The forces of the non-augmented particles before any augmentation is done.
        Note that augmentation modifies the forces on the real particles in the
        extended phase space, but that is not reflected in this entry (but is
        reflected in forces).
    copy:
        Copies the object without refreshing.
    pullback:
        Given a callable that acts on an AugementedTrajectory instance creates
        a callable that acts on a Trajectory instance or on array data.
    from_trajectory:
        Alternate init that uses a Trajectory object.
    __getitem__:
        Subsets the instance without refreshing. Only slices are allowed.

    """

    def __init__(
        self,
        coords: ndarray,
        forces: ndarray,
        augmenter: Augmenter,
        kbt: float,
        override_first_augment: Optional[Tuple[ndarray, ndarray]] = None,
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        coords:
            coordinates representing the real part of the system obtained from an
            external method (e.g., simulation). Should have shape
            `(n_frames,n_sites,n_dims)` (See parent class).
        forces:
            forces (thermally scaled density log gradients) representing the real part
            of the system obtained from an external method (e.g. simulation).
        augmenter:
            Transform that is used extend trajectory. Must be of the Augmenter class.
        kbt:
            Thermal scaling of the molecular system. Used to linearly scale the log
            derivatives given by augmenter. Should be the same unit as the energy
            component of forces.
        override_first_augment:
            If not None, the content of this option is used to create the initial
            augmented force and coordinate entry. It must then be a two member
            collection where the first element is the augmented coordinates and the
            second entry is the augmented forces.

            This argument is mainly for internal use in slicing. It is unlikely to be
            applicable to normal usage.
        """
        self.augmenter = augmenter
        self.kbt = kbt
        # forces are served via an attribute.
        self._real_forces = forces
        self._real_n_sites = coords.shape[1]
        # if augmented_coords/forces are passed, use them. Otherwise generate them.
        if override_first_augment is None:
            ext_coords, ext_forces = self._augment(coords, forces)
        else:
            ext_coords, ext_forces = override_first_augment
        # create trajectory using augmented coordinates and forces
        super().__init__(coords=ext_coords, forces=ext_forces)

    def _augment(self, coords: ndarray, forces: ndarray) -> Tuple[ndarray, ndarray]:
        """Create an instance of augmented coordinates and forces.

        This isn't public because the more universal approach is to call .refresh
        and then use the changed attributes.

        Arguments:
        ---------
        coords:
            Array of shape (n_frames,n_sites,n_dims) containing the positions of the
            real particles.  N_sites corresponds to the number of real particles.
        forces:
            Array of shape (n_frames,n_sites,n_dims) containing the forces of the
            real particles.  Must be the same shape as coords.

        Returns:
        -------
        Tuple of arrays: first element is the combination of the real and augmenting
        coordinates, and the second element is the combination of the second and

        Note:
        ----
        Note that this process modifies the forces corresponding to original
        positions in the output. See class description. Additionally, note
        that self.kbt is used to scale the log derivative output of augmenter.

        """
        aug_coords = self.augmenter.sample(coords)
        real_lgrad_correction, aug_lgrad = self.augmenter.log_gradient(
            coords, aug_coords
        )
        aug_forces = self.kbt * aug_lgrad
        real_forces_corrected = forces + self.kbt * real_lgrad_correction
        full_coords = concatenate([coords, aug_coords], axis=1)
        full_forces = concatenate([real_forces_corrected, aug_forces], axis=1)
        return (full_coords, full_forces)

    @property
    def real_coords(self) -> ndarray:
        """Return the coordinates of the real particles."""
        return self.coords[:, : self._real_n_sites, :]

    @real_coords.setter
    def real_coords(self, value: Any) -> NoReturn:  # noqa: ARG002
        """Real positions cannot be set directly."""
        raise ValueError("real_positions cannot be reassigned.")

    @property
    def real_forces(self) -> ndarray:
        """Return the coordinates of the real particles.

        Note that this entry's forces are as if no augmentation had been performed,
        and will not match the corresponding entries in the forces attribute.
        """
        return self._real_forces

    @real_forces.setter
    def real_forces(self, value: Any) -> NoReturn:  # noqa: ARG002
        """Real forces cannot be set directly."""
        raise ValueError("real_forces cannot be reassigned.")

    @property
    def n_real_sites(self) -> int:
        """Number of real particles in the system."""
        return self.real_coords.shape[1]

    @property
    def n_aug_sites(self) -> int:
        """Number of augmenting particles in the system."""
        return self.coords.shape[1] - self.real_coords.shape[1]

    def refresh(
        self,
    ) -> None:
        """Refresh the coords and forces attributes by reapplying augmenter."""
        new_coords, new_forces = self._augment(
            coords=self.real_coords, forces=self.real_forces
        )
        # this resets the _augmented_ coordinate entries.
        self.coords = new_coords
        self.forces = new_forces

    def __getitem__(self, index: slice) -> "AugmentedTrajectory":
        """Index trajectory without refreshing augmentation.

        Only slices are allowed. Returns a AugmentedTrajectory instance.
        """
        if not isinstance(index, slice):
            raise ValueError("Only slices are allowed for indexing.")
        # these are the full noise+real entries
        new_aug_coords = self.coords[index]
        new_aug_forces = self.forces[index]
        # we use the override argument so that we don't do another
        # random redraw
        return AugmentedTrajectory(
            coords=self.real_coords[index],
            forces=self.real_forces[index],
            augmenter=self.augmenter,
            kbt=self.kbt,
            override_first_augment=(new_aug_coords, new_aug_forces),
        )

    def copy(self) -> "AugmentedTrajectory":
        """Copy trajectory without refreshing augmentation."""
        real_coords_copy = self.real_coords.copy()
        real_forces_copy = self.real_forces.copy()
        aug_coords_copy = self.coords.copy()
        aug_forces_copy = self.forces.copy()
        augmenter_copy = deepcopy(self.augmenter)
        ob_copy = self.__class__(
            coords=real_coords_copy,
            forces=real_forces_copy,
            augmenter=augmenter_copy,
            kbt=self.kbt,
            override_first_augment=(aug_coords_copy, aug_forces_copy),
        )
        return ob_copy

    @overload
    def pullback(
        self, C: Callable[["AugmentedTrajectory"], A], array: Literal[True]
    ) -> Callable[[ndarray, ndarray], A]:
        ...

    @overload
    def pullback(
        self, C: Callable[["AugmentedTrajectory"], A], array: Literal[False]
    ) -> Callable[[Trajectory], A]:
        ...

    def pullback(
        self,
        C: Callable[["AugmentedTrajectory"], A],
        array: bool = False,
    ) -> Callable:
        """Pull back a callable on this instance to to initialization arguments.

        Given callable `C` operating on `A`, where `A`  is a
        AugmentedTrajectory instance, this method returns a new callable `D` that
        roughly works as follows:
        ```
        def D(data,*args,*kwargs):
            A = AugmentedTraj(data, self.augmenter, self.kbt)
            return C(A)
        ```
        That is, it creates a callable that first creates an
        `AugmentedTrajectory` and then applies the provided callable. The
        `AugmentedTrajectory` is created using both information provided by the
        outer callable and information from the current instance (e.g., `augmenter`).

        Arguments:
        ---------
        C:
            Callable to be pulled back. Should accept an argument of type
            `AugmentedTrajectory`.
        array:
            If true, the produced callable will take two arguments: the first
            is a coords `ndarray` and the second is a forces `ndarray`. If false,
            the produced callable will take a Trajectory instance as its only
            argument. In both cases an intermediate AugmentedTrajectory is made.

        Return:
        ------
        Either a callable with signature Callable[[ndarray,ndarray],A] or
        Callable[[Trajectory],A]. See `array` argument.
        """
        if array:

            def array_wrapped(coords: ndarray, forces: ndarray) -> A:
                at = self.__class__(
                    coords=coords, forces=forces, augmenter=self.augmenter, kbt=self.kbt
                )
                return C(at)

            return array_wrapped

        else:

            def traj_wrapped(t: Trajectory) -> A:
                at = self.__class__(
                    coords=t.coords,
                    forces=t.forces,
                    augmenter=self.augmenter,
                    kbt=self.kbt,
                )
                return C(at)

            return traj_wrapped

    @classmethod
    def from_trajectory(
        cls, t: Trajectory, kbt: float, augmenter: Augmenter
    ) -> "AugmentedTrajectory":
        """Create AugmentedTrajectory from a Trajectory object.

        Additional initialization information is passes via arguments.

        Arguments:
        ---------
        t:
            Trajectory object to transform into AugmentedTrajectory. Note
            that coord and forces entries are transferred without copy.
        kbt:
            kbt object. Passed to class init.
        augmenter:
            Augmenter object. Passed to class init.

        Returns:
        -------
        AugmentedTrajectory object

        """
        return cls(
            coords=t.coords,
            forces=t.forces,
            augmenter=augmenter,
            kbt=kbt,
        )
