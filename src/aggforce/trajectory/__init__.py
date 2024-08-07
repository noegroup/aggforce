"""Provides tools and definitions of Trajectory instances."""
# __init__ doesn't use the imported objects
# ruff: noqa: F401
from .core import (
    ForcesTrajectory,
    CoordsTrajectory,
    Trajectory,
    AugmentedTrajectory,
)
from .augment import Augmenter

try:
    from .jaxgausstraj import JCondNormal
except ImportError:
    pass
