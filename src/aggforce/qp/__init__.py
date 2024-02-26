"""Provides routines for quadratic programming map optimization."""
# __init__ doesn't use the imported objects
# ruff: noqa: F401
from .qplinear import (
    qp_linear_map,
    qp_form,
    make_bond_constraint_matrix,
)
from .basicagg import constraint_aware_uni_map
from .featlinearmap import (
    FeatZipper,
    Multifeaturize,
    GeneralizedFeatures,
    GeneralizedFeaturizer,
    qp_feat_linear_map,
    id_feat,
)

try:
    from .jaxfeat import gb_feat
    from .jgauss import joptgauss_map
except ImportError:
    pass
