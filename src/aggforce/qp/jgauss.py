"""Provides jax methods for making optimized stochastic coordinate-force maps."""

from typing import Optional
import warnings
import numpy as np
from ..map import (
    LinearMap,
    JLinearMap,
    AugmentedTMap,
    SeperableTMap,
    NullForcesTMap,
    lmap_augvariables,
    ComposedTMap,
    RATMap,
)
from ..trajectory import (
    Trajectory,
    CoordsTrajectory,
    AugmentedTrajectory,
    JCondNormal,
)
from ..constraints import Constraints
from .qplinear import qp_linear_map, DEFAULT_SOLVER_OPTIONS, SolverOptions
from .basicagg import constraint_aware_uni_map


def joptgauss_map(
    traj: Trajectory,
    coord_map: LinearMap,
    var: float,
    kbt: float,
    constraints: Optional[Constraints] = None,
    seed: Optional[int] = None,
    **kwargs,
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
    # however, it needs to in the form of jax function which acts on flattened
    # vectors and allows for single-frame operation. This is accessible via an
    # attributed of a jaxxed LinearMap (JLinearMap) attribute.
    flattened_cmap = JLinearMap.from_linearmap(
        coord_map, bypass_nan_check=True
    ).flat_call
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


def stagedjoptgauss_map(
    traj: Trajectory,
    coord_map: LinearMap,
    var: float,
    kbt: float,
    force_map: Optional[LinearMap] = None,
    constraints: Optional[Constraints] = None,
    seed: Optional[int] = None,
    premap_l2_regularization: float = 0.0,
    premap_solver_args: SolverOptions = DEFAULT_SOLVER_OPTIONS,
    **kwargs,
) -> ComposedTMap:
    """Create optimized Gaussian map with linear premap and subsequent noise step.

    For a simpler Gaussian map without explicit premap, see joptgauss_map.

    This routine offers the advantage that the first submap of the returned map
    may be used to process the data before saving; then, during subsequent use,
    this partially mapped data may be loaded and transformed using the second
    map

    This routine performs the following steps:
        1. Generate an optimized force map without any noise.
            - Note: if force_map is specified, this is used in lieu of 1.
        2. Create a augmented trajectory without any premap.
        3. (Partially) map the real particles augmented trajectory using the non-noise
           optimized map.
        4. Create an optimized map on the partially mapped augmented trajectory.
        5. Compose the maps from 1 and 4 to create a new map.

    To access the premap, index the returned TMap with [1]. To obtain the noise
    map, index it with [0].

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
    force_map:
        If not None, this is used instead of performing the initial no-noise force map
        optimization.
    constraints:
        Molecular constraints of present in traj's simulator. Used in force map design.
    seed:
        Random seed that will be passed to the Gaussian noiser (JCondNormal instance).
    premap_l2_regularization:
        l2_regularization passed to initial non-noised force map optimization
        (qp_linear_map).
    premap_solver_args:
        Arguments passed to initial non-noised force map optimization (qp_linear_
    **kwargs:
        Passed to underlying qp_linear_map optimization on the derived
        AugmentedTrajectory.

    Returns:
    -------
    An ComposedTMap which characterizes the Gaussian map. This map has two submaps; the
    first is a deterministic map that coarse-grains the coordinates and forces,
    and the second map applies noising operations. The data map mapped with the first
    map, saved, loaded, and then mapped with the second map.

    """
    # first create non-noised optimized force map
    if force_map is None:
        pre_tmap = qp_linear_map(
            traj=traj,
            coord_map=coord_map,
            constraints=constraints,
            l2_regularization=premap_l2_regularization,
            solver_args=premap_solver_args,
        )
    else:
        pre_tmap = SeperableTMap(coord_map=coord_map, force_map=force_map)

    # We then extract the noise and coord maps and jaxify them.
    #
    # we know based on external knowledge that these entrees are LinearMaps
    j_coord_map = JLinearMap.from_linearmap(pre_tmap.coord_map, bypass_nan_check=True)  # type: ignore [arg-type]
    j_force_map = JLinearMap.from_linearmap(pre_tmap.force_map, bypass_nan_check=True)  # type: ignore [arg-type]

    # We then create the augmenter. This will be used with the full trajectory.
    augmenter = JCondNormal(cov=var, premap=j_coord_map.flat_call, seed=seed)
    # This trajectory contains all the source atoms and the noise sizes.
    aug_traj = AugmentedTrajectory.from_trajectory(t=traj, augmenter=augmenter, kbt=kbt)
    # We map this trajectory to only have the noise sizes AND the coarse-grained version
    # of the real sites. This allows us to map the noise forces on the real sites using
    # the pre-derived force map.
    pmapped_traj = RATMap(tmap=pre_tmap)(aug_traj)

    # we now work towards maping a force map on pmapped_traj.

    # pmapped* is a Trajectory, not an AugmentedTrajectory. So we must manually
    # create the coordinate map for the second optimization. This coordinate
    # map isolates the noise particles, similar to that created by lmap_augvariables.
    preserved_sites = []
    for index in range(
        pmapped_traj.n_sites - aug_traj.n_aug_sites, pmapped_traj.n_sites
    ):
        preserved_sites.append([index])
    pmapped_coord_map = LinearMap(
        mapping=preserved_sites, n_fg_sites=pmapped_traj.n_sites
    )

    # we then move to creating the force map.  we no longer know what the
    # constraints are (they have probably been mapped away). For a reasonable
    # pre-coord map, there shouldn't be any left, and we assume this is true.
    pmapped_tmap = qp_linear_map(
        traj=pmapped_traj, coord_map=pmapped_coord_map, constraints=set(), **kwargs
    )

    # we now create the composed map.
    #
    # (j_force_map @ j_coord_map.T) is an important object. It is the
    # JLinearMap given given by the standard_matrix in both instances being
    # appropriately multiplied.  This can be see by first noting that \grad_x
    # f(A x) = A^T [\grad f] (A x); in our application, j_coord_map corresponds
    # to A, and [\grad f] is the force calculated w.r.t. the CG particles. A^T
    # [\grad f] (h) therefore is an expression for the atomistic forces given
    # by the noise when only the coarse-grained positions are provided (h).
    # This expression is then mapped via j_force_map.
    #
    # Collectively, (j_force_map @ j_coord_map.T) transforms the CG real
    # coordinate forces given by the Augmenter into mapped atomistic forces. As
    # a result, the following Augmenter now corrects the forces in a
    # _already mapped_ atomistic trajectory.

    pmapped_augmenter = JCondNormal(
        cov=var,
        source_postmap=(j_force_map @ j_coord_map.T),
        seed=seed,
    )

    post_tmap = AugmentedTMap(
        aug_tmap=pmapped_tmap,
        augmenter=pmapped_augmenter,
        kbt=kbt,
    )

    # comb_tmap, when applied to a source trajectory will perform the following steps:
    # 1. Apply pre_tmap
    #       The incoming data is not yet coarse-grained. pre_tmap will use the derived
    #       linear force and coordinate maps to map the data to the CG resolution.
    #           X = j_coord_map(x) #noqa
    #           Y = j_force_map(y) #noqa
    #                           ^ y is the atomistic force, x the atomistic coords.
    # 2. Apply post_tmap
    #           X_final = X+[0-mean gaussian noise]
    #           Y_final = Y+[(j_force_map @ j_coord_map.T)([noise-force])] #noqa
    #       note that by substitution, linearity, and @, we get
    #           Y_final = j_force_map(y + j_coord_map.T*[noise-force]) #noqa
    #                                             ^this is the "backmapped" noise force
    #                                     ^which is then combined with the atomistic
    #                                      force and mapped

    comb_tmap = ComposedTMap(submaps=[post_tmap, pre_tmap])

    return comb_tmap


def stagedjslicegauss_map(
    traj: CoordsTrajectory,
    coord_map: LinearMap,
    var: float,
    kbt: float,
    seed: Optional[int] = None,
    constraints: Optional[Constraints] = None,  # noqa: ARG001
    warn_input_forces: bool = True,
) -> ComposedTMap:
    """Create Gaussian map which only uses information from noising in reported forces.

    This routine is written to mirror the procedure in stagedjoptgauss_map, and
    similarly outputs a ComposedTMap; however, this ComposedTMap has 3 parts.
    maps[2] adds null forces to the input data if needed (allowing the derived
    tmap to operate when no forces are present), maps[1] maps the coordinates to the
    coarse-grained resolution, and maps[0] noises the data and extracts noise-derived
    forces.

    At the cost of increasing complexity, we keep the internal procedure close to
    that in stagedjoptgauss_map. The following steps are performed:
        1. Set forces in input to nans to make sure they are not used.
        2. Create a augmented trajectory without any premap.
        3. Partially map the augmented trajectory to the resolution of mapped real
           sites with augmented sites.
        4. Create a slice force map on the partially mapped trajectory.
        5. Compose the maps from 1, 3, and 4 to create a new map.

    Arguments:
    ---------
    traj:
        Trajectory instance that will be used to create noised positions then
        subsequently mapped.
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
    seed:
        Random seed that will be passed to the Gaussian noiser (JCondNormal instance).
    constraints:
        Not used. Retained for compatibility.
    warn_input_forces:
        If True, we warn if forces were provided in the input data, as we will ignore
        them.

    Returns:
    -------
    An ComposedTMap which characterizes the Gaussian map. This map has three
    submaps; the first map adds dummy forces if the input data lacks forces.
    The second map reduces the dimension of the data via coord_map, and the
    third map noises the system and isolates the noise-derived forces.

    """
    # to be sure that we do not actually use force information present in the input
    # trajectory, we replace it with NaNs. This also allows the input to
    # not have force information without changing subsequent calls.
    naforce_traj = NullForcesTMap(warn_input_forces=warn_input_forces)(traj)

    # Create augmenter that adds gaussian noise to the system.
    # bypass_nan_check is needed for internal derivative calculations.
    augmenter = JCondNormal(
        cov=var,
        premap=JLinearMap.from_linearmap(coord_map, bypass_nan_check=True).flat_call,
        seed=seed,
    )
    # create augmented trajectory
    aug_traj = AugmentedTrajectory.from_trajectory(
        t=naforce_traj, augmenter=augmenter, kbt=kbt
    )

    # we now create the partially mapped augmented trajectory, but unlike in
    # other methods we must create a dummy force map, and then use that to
    # create a preprocessing tmap with coord_map.
    null_fmap = LinearMap(
        mapping=np.ones_like(coord_map.standard_matrix), handle_nans=False
    )
    pre_tmap = SeperableTMap(coord_map=coord_map, force_map=null_fmap)

    # this contains the noise particles and mapped real particles.
    pmapped_traj = RATMap(tmap=pre_tmap)(aug_traj)

    # create the map that isolates the noise sites on the partially mapped traj.
    preserved_sites = []
    for index in range(
        pmapped_traj.n_sites - aug_traj.n_aug_sites, pmapped_traj.n_sites
    ):
        preserved_sites.append([index])
    pmapped_coord_map = LinearMap(
        mapping=preserved_sites, n_fg_sites=pmapped_traj.n_sites
    )

    # we then move to creating the force map that acts on the partially mapped
    # traj.  We no longer know what the constraints are (they have probably
    # been mapped away). For a reasonable pre-coord map, there shouldn't be any
    # left, and we assume this is true.
    pmapped_tmap = constraint_aware_uni_map(
        traj=pmapped_traj,
        coord_map=pmapped_coord_map,
        constraints=set(),
    )

    # this is the augmenter that acts on the already coarse-grained traj.
    # As we do not use any forces on the real particles, we do not bother
    # to create a force-modifier as is done in other methods.
    pmapped_augmenter = JCondNormal(
        cov=var,
        seed=seed,
    )

    # we wrapped the derived force map with the augmentation operation
    post_tmap = AugmentedTMap(
        aug_tmap=pmapped_tmap,
        augmenter=pmapped_augmenter,
        kbt=kbt,
    )

    # and finally compose maps to create the returned callable.
    # NullForcesTMap allows the resulting TMap to be applied to trajectories
    # which do not have force information, or coordinate arrays.
    comb_tmap = ComposedTMap(
        submaps=[post_tmap, pre_tmap, NullForcesTMap(warn_input_forces=False)]
    )

    return comb_tmap


def stagedjforcegauss_map(
    traj: Trajectory,
    coord_map: LinearMap,
    var: float,
    kbt: float,
    force_map: Optional[LinearMap] = None,
    constraints: Optional[Constraints] = None,
    seed: Optional[int] = None,
    premap_l2_regularization: float = 0.0,
    premap_solver_args: SolverOptions = DEFAULT_SOLVER_OPTIONS,
    contribution_tolerance: float = 1e-6,
    **kwargs,
) -> ComposedTMap:
    """Create source-force-only Gaussian map with linear premap and subsequent noising.

    This routine creates a trajectory map that noises coordinates, and then introduces
    a minimal amount of noise-derived force information into the induced force signal.
    For many conditions the level of noise-derived force may be brought to zero.

    This routine performs the following steps:
        1. Generate an optimized force map without any noise.
            - Note: if force_map is specified, this is used in lieu of 1.
        2. Create a augmented trajectory without any premap.
        3. (Partially) map the real particles augmented trajectory using the non-noise
           optimized map.
        4. Create an optimized map on the partially mapped augmented trajectory.
            - Critically, this map is optimized not to reduce all force noise but reduce
                noise-only force noise.
        5. Compose the maps from 1 and 4 to create a new map.

    To access the premap, index the returned TMap with [1]. To obtain the noise
    map, index it with [0]. Note that due to the nature of the extended ensemble,
    the force-map will still mix forces from the virtual and real particles; however,
    the resulting force should be close to the real-particle force present before
    creating the extended ensemble.

    This method is structured to mirror stagedjoptgauss_map.

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
        turn the log density gradients of the added noise variates into forces. If all
        force-related contributions are removed, this should not affect mapped forces.
    force_map:
        If not None, this is used instead of performing the initial no-noise force map
        optimization.
    constraints:
        Molecular constraints present in traj's simulator. Used in force map design.
    seed:
        Random seed that will be passed to the Gaussian noiser (JCondNormal instance).
    premap_l2_regularization:
        l2_regularization passed to initial non-noised force map optimization
        (qp_linear_map).
    premap_solver_args:
        Arguments passed to initial non-noised force map optimization.
    contribution_tolerance:
        We check the mean l2-norm of the noise-derived force contribution over the given
        trajectory and compare it to this value; if it is larger, we warn.
    **kwargs:
        Passed to underlying qp_linear_map optimization on the derived
        AugmentedTrajectory.

    Returns:
    -------
    An ComposedTMap which characterizes the Gaussian map. This map has two submaps; the
    first is a deterministic map that coarse-grains the coordinates and forces,
    and the second map applies noising operations. Data may be mapped with the first
    map, saved, loaded, and then mapped with the second map.

    """
    # first create non-noised optimized force map
    if force_map is None:
        pre_tmap = qp_linear_map(
            traj=traj,
            coord_map=coord_map,
            constraints=constraints,
            l2_regularization=premap_l2_regularization,
            solver_args=premap_solver_args,
        )
    else:
        pre_tmap = SeperableTMap(coord_map=coord_map, force_map=force_map)

    # We then extract the noise and coord maps and jaxify them.
    #
    # we know based on external knowledge that these entrees are LinearMaps
    j_coord_map = JLinearMap.from_linearmap(pre_tmap.coord_map, bypass_nan_check=True)  # type: ignore [arg-type]
    j_force_map = JLinearMap.from_linearmap(pre_tmap.force_map, bypass_nan_check=True)  # type: ignore [arg-type]

    # We then create the augmenter. This will be used with the full trajectory.
    augmenter = JCondNormal(cov=var, premap=j_coord_map.flat_call, seed=seed)

    # When we optimize the second-resolution map, we want to miniminize only
    # the noise contributions.  In order to do that we remove all real-force
    # contributions and then then optimize the remaining forces as normal.
    # zeroforce_traj zeros out the trajectory force contributions.
    zeroforce_traj = Trajectory(coords=traj.coords, forces=np.zeros_like(traj.forces))

    # This trajectory contains all the source atoms and noise sites, and only
    # has noise-force contributions.
    aug_traj = AugmentedTrajectory.from_trajectory(
        t=zeroforce_traj, augmenter=augmenter, kbt=kbt
    )

    # We map this trajectory to only have the noise sites AND the coarse-grained version
    # of the real sites. This allows us to map the noise forces on the real sites using
    # the pre-derived force map.
    pmapped_traj = RATMap(tmap=pre_tmap)(aug_traj)

    # we now work towards optimizing a force map on pmapped_traj that minimizes
    # noise-force contributions.

    # pmapped* is a Trajectory, not an AugmentedTrajectory. So we must manually
    # create the coordinate map for the second optimization. This coordinate
    # map isolates the noise particles, similar to that created by lmap_augvariables.
    preserved_sites = []
    for index in range(
        pmapped_traj.n_sites - aug_traj.n_aug_sites, pmapped_traj.n_sites
    ):
        preserved_sites.append([index])
    pmapped_coord_map = LinearMap(
        mapping=preserved_sites, n_fg_sites=pmapped_traj.n_sites
    )

    # we then move to creating the force map.  we no longer know what the
    # constraints are (they have probably been mapped away). For a reasonable
    # pre-coord map, there shouldn't be any left, and we assume this is true.
    pmapped_tmap = qp_linear_map(
        traj=pmapped_traj, coord_map=pmapped_coord_map, constraints=set(), **kwargs
    )
    # this derived force map is treated as a general force map that may have noise
    # contributions.

    # we check how big the noise contributions are
    remaining_force_residual = np.mean(pmapped_tmap(pmapped_traj).forces ** 2)
    if remaining_force_residual > contribution_tolerance:
        warnings.warn(
            "Unable to remove all noise contributions in forces. Remaining "
            f"contribution: {remaining_force_residual}.",
            stacklevel=0,
        )

    # we now create the composed map.
    #
    # (j_force_map @ j_coord_map.T) is an important object. It is the
    # JLinearMap given given by the standard_matrix in both instances being
    # appropriately multiplied.  This can be see by first noting that \grad_x
    # f(A x) = A^T [\grad f] (A x); in our application, j_coord_map corresponds
    # to A, and [\grad f] is the force calculated w.r.t. the CG particles. A^T
    # [\grad f] (h) therefore is an expression for the atomistic forces given
    # by the noise when only the coarse-grained positions are provided (h).
    # This expression is then mapped via j_force_map.
    #
    # Collectively, (j_force_map @ j_coord_map.T) transforms the CG real
    # coordinate forces given by the Augmenter into mapped atomistic forces. As
    # a result, the following Augmenter now corrects the forces in a
    # _already mapped_ atomistic trajectory.

    pmapped_augmenter = JCondNormal(
        cov=var,
        source_postmap=(j_force_map @ j_coord_map.T),
        seed=seed,
    )

    post_tmap = AugmentedTMap(
        aug_tmap=pmapped_tmap,
        augmenter=pmapped_augmenter,
        kbt=kbt,
    )

    # comb_tmap, when applied to a source trajectory will perform the following steps:
    # 1. Apply pre_tmap
    #       The incoming data is not yet coarse-grained. pre_tmap will use the derived
    #       linear force and coordinate maps to map the data to the CG resolution.
    #           X = j_coord_map(x) #noqa
    #           Y = j_force_map(y) #noqa
    #                           ^ y is the atomistic force, x the atomistic coords.
    # 2. Apply post_tmap
    #           X_final = X+[0-mean gaussian noise]
    #           Y_final = Y+[(j_force_map @ j_coord_map.T)([noise-force])] #noqa
    #       note that by substitution, linearity, and @, we get
    #           Y_final = j_force_map(y + j_coord_map.T*[noise-force]) #noqa
    #                                             ^this is the "backmapped" noise force
    #                                     ^which is then combined with the atomistic
    #                                      force and mapped

    comb_tmap = ComposedTMap(submaps=[post_tmap, pre_tmap])

    return comb_tmap
