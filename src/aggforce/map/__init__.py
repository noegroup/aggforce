"""Provides tools and definitions for maps."""
# __init__ doesn't use the imported objects
# ruff: noqa: F401
from .core import LinearMap, CLAMap, trjdot
from .tmap import TMap, SeperableTMap, CLAFTMap, AugmentedTMap, ComposedTMap, RATMap
from .tools import lmap_augvariables, smear_map

# in case jax is not installed
try:
    from .jaxtools import jaxify_linearmap
    from .jaxlinearmap import JLinearMap
except ImportError:
    pass
