"""Provides jax methods for making optimized stochastic coordinate-force maps."""

from typing import Optional
from ..map import LinearMap, AugmentedTMap, jaxify_linearmap, lmap_augvariables
from ..trajectory import Trajectory, AugmentedTrajectory, JCondNormal
from ..constraints import Constraints
from .qplinear import qp_linear_map


def joptgauss_map(
    traj: Trajectory,
    coord_map: LinearMap,
    var: float,
    kbt: float,
    constraints: Optional[Constraints] = None,
    seed: Optional[int] = None,
    **kwargs
) -> AugmentedTMap:
    """Create optimized Gaussian map.

    Gaussian maps are stochastic Trajectory maps that add Gaussian noise to
    coordinates and subsequently combine force information derived from this noise
    with forces present in the input Trajectory. The derived map operates solely
    on Trajectory objects, hiding the intermediate addition of noise and subsequent
    internal mapping operations.

    The level of mixing between reference and noise derived force information is
    optimized to minimize the mean force norm during training.

    Note that unlike some other routines, the returned TMap instance is not
    separable, and so cannot be used to map coords and forces arrays
    independently. Either create a Trajectory object as input for mapping or
    use the `map_arrays` method to map a pair of coords and forces.
    Furthermore, note that the derived mapping is not deterministic--- mapping
    a given trajectory twice will produce two different mapped Trajectories,
    each drawn from a fixed probability distribution.

    This routine does roughly the following steps:
        1. Create an AugmentedTrajectory:
            The first n particles of this instance are the particles present in traj.
            N new particles are added by adding Gaussian noise to the system and
            treating the new output as a new particle position. This noise is added
            to the system after `coord_map` is applied.

            For example, if coord_map isolates the carbon-alphas of the system,
            Gaussian noise is added to the carbon alphas to create "noised carbon-
            alphas" which augment the state space of the protein. The full phase
            space is all of the sites in traj and the noised carbon-alphas.

        2.  Create an force map (LinearMap instance) that maps the forces in this
            AugmentedTrajectory using qp_linear_map. This map mixes forces from
            noise and those from traj and is made with respect to a new config_map
            that isolates the particles introduced via the addition of noise.

        3.  The augmented force map  is combined with the new coord map (not
            the original coord_map) and wrapped to create a TMap that operates on a
            non-augmented input trajectory: internally it takes its input, augments
            it, and then applies the generated extended maps. The dimensionality of
            this output matches that which would have been created the original
            coord_map.

    Arguments:
    ---------
    traj:
        Trajectory instance that will be used to create the optimized force map and
        then subsequently mapped.
    coord_map:
        Coordinate map representing the coarse-grained description of the system. The
        output dimension (n_cg_sites) determines the number of auxiliary particles to
        the Gaussian noise augmenter will add to the system.

        Note that this map does not enter the produced TMap in a straightforward way.
    var:
        The noise added is drawn from a Gaussian with a diagonal covariance matrix; this
        positive scalar is the diagonal value. A small value means the level of noise
        added is small, and larger values perturb the system more.
    kbt:
        Boltzmann constant times temperature for the samples in traj. This is needed to
        turn the log density gradients of the added noise variates into forces.
    constraints:
        Molecular constraints of present in traj's simulator. Used in force map design.
    seed:
        Random seed that will be passed to the Gaussian noiser (JCondNormal instance).
    **kwargs:
        Passed to underlying qp_linear_map optimization on the derived
        AugmentedTrajectory.

    Returns:
    -------
    An AugmentedTMap which characterizes the Gaussian map.

    """
    # the coord_map is used in the definition of JCondNormal, which does the noising.
    # however, it needs to be modified into a jax function which acts on flattened
    # vectors and allows for single-frame operation.
    flattened_cmap = jaxify_linearmap(coord_map)
    # create the object that will do the noising
    augmenter = JCondNormal(cov=var, premap=flattened_cmap, seed=seed)
    # create extended trajectory using the derived noiser
    aug_traj = AugmentedTrajectory.from_trajectory(t=traj, augmenter=augmenter, kbt=kbt)
    # this is the true coord map that acts on the AugmentedTrajectory to isolate the
    # generated particles
    aug_coord_map = lmap_augvariables(aug_traj)
    # Optimize a linear force map on the augmented trajectory.
    #
    # we are utilizing the fact that the constraints are denoted via indices, and those
    # indices still refer to the same real sites in the AugmentedTrajectory as the
    # generated particles are placed at the end of the trajectory.
    aug_tmap = qp_linear_map(
        traj=aug_traj, coord_map=aug_coord_map, constraints=constraints, **kwargs
    )

    # make wrapped trajectory map that first augments and then applies the derived map
    tmap = AugmentedTMap(
        aug_tmap=aug_tmap,
        augmenter=augmenter,
        kbt=kbt,
    )

    return tmap
