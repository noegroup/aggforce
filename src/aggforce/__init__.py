"""Maps forces for coarse-graining molecular dynamics trajectories.

Coarse-grained force-fields can be created by matching the forces produced from a higher
resolution simulation. This process requires the higher resolution forces be mapped.
This module provides routines for performing this mapping in different ways, but does
not itself parameterize any force-fields.

The primary entry point is project_forces. Only basic routines for simple linear map 
optimization are visible in the module namespace; more advanced features require 
explicit imports from submodules.

See agg.py and the README for more information. Tests and examples are also available.
"""

# __init__ doesn't use the imported objects
# ruff: noqa: F401
from .trajectory import Trajectory
from .agg import project_forces
from .constraints import guess_pairwise_constraints
from .qp import qp_linear_map, constraint_aware_uni_map
from .map import LinearMap

# in case jax is not installed
try:
    from .qp import joptgauss_map
except ImportError:
    pass
