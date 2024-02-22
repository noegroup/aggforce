"""Provides tools and definitions for maps."""
# __init__ doesn't use the imported objects
# ruff: noqa: F401
from .core import LinearMap, CLAMap, trjdot, smear_map
from .tmap import TMap, SeperableTMap, CLAFTMap, AugmentedTMap

# in case jax is not installed
try:
    from .jaxtools import jaxify_linearmap
except ModuleNotFoundError:
    pass
