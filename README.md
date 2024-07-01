# aggforce

A package to aggregate atomistic forces to estimate the forces of a given
manybody potential of mean force. 

### Installation

Install the aggforce package from source by calling `pip install .`,  
`pip install .[nonlinear]`, or `pip install .[test]` from the 
repository's root directory. The initial option will install only what is 
needed to optimize linear maps, the second will install what is needed for 
featurized (nonlinear) maps, and the third allows one to run `pytest`.

In order to use the built-in featurizers to find configurationally dependent
force mappings, [JAX](https://github.com/google/jax) must be installed. 
The `[nonlinear]` and `[test]` targets satisfy this using a CPU accelerated 
version of JAX.  However, it is often necessary to install a GPU 
accelerated version; for instructions on how to do so, see the JAX 
[documentation](https://github.com/google/jax).

### Example usage

The following code shows how to generate an optimal linear force aggregation
map that does not change based on molecular configuration. We grab test data, 
create a carbon alpha configurational mapping, detect
constrained bonds from the trajectory, and then produce and apply an optimized
force aggregation map to the trajectory.

```python
from aggforce import (LinearMap, 
                      guess_pairwise_constraints, 
		      project_forces, 
		      constraint_aware_uni_map,
		      )
import numpy as np
import re
import mdtraj as md

# get data
forces = np.load("tests/data/cln025_record_2_prod_97.npz")["Fs"]
coords = np.load("tests/data/cln025_record_2_prod_97.npz")["coords"]
pdb = md.load("tests/data/cln025.pdb")

# we use a carbon alpha configurational map, so we use mdtraj to get a topology an
# then filter by name to get a map.  The map is of the form
# [[inds1],[inds2],[inds3] where each list element of the parent list corresponds
# to the atoms contributing to a particular cg particle

inds = []
atomlist = list(pdb.topology.atoms)
for ind, a in enumerate(atomlist):
    if re.search(r"CA$", str(a)):
        inds.append([ind])

# linear transformations (for forces and configurations) are represented by 
# LinearMap instances

# we create our configurational c-alpha map, which is needed to optimize
# the force map
cmap = LinearMap(inds, n_fg_sites=coords.shape[1])

# detect which atoms have bond constraints based on statistics, only use 10
# frames
constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)
# get force map which uniformly aggregates forces inside the cg bead and adds
# other atoms to satisfy constraint rules
basic_results = project_forces(
    xyz=None,
    forces=forces,
    config_mapping=cmap,
    constrained_inds=constraints,
    method=constraint_aware_uni_map,
)
# get _optimized_ force map which optimally weights atoms' forces for
# aggregation
optim_results = project_forces(
    xyz=None, forces=forces, config_mapping=cmap, constrained_inds=constraints
)

# optim_results and basic_results are dictionaries full of the results

# optimal map itself optim_results['map']
#   this object is callable on mdtraj formatted force/position arrays and maps
#   them
# optimal map _matrix_ is optim_results['map'].standard_matrix
# forces processed via the optimal map are under optim_results['project_forces']
# similarly for basic map
```

Optimized force mappings which are allowed to change as a function of
configuration can be created as follows. However, first note that this approach
depends on features: these features control how the map can change as a function
of configuration.  Second, note that JAX must be installed to use the features
included in this library (as we do here). Finally, note that this approach is
much more computationally expensive than the static mappings and has not yet
been shown to produce significantly better results.

```python
from aggforce import (LinearMap, 
                      guess_pairwise_constraints, 
		      project_forces, 
		      constraint_aware_uni_map,
		      qp_feat_linear_map,
		      )
from aggforce.util import Curry
from aggforce.qp import Multifeaturize, gb_feat, id_feat
import numpy as np
import re
import mdtraj as md

forces = np.load("tests/data/cln025_record_2_prod_97.npz")["Fs"]
coords = np.load("tests/data/cln025_record_2_prod_97.npz")["coords"]
pdb = md.load("tests/data/cln025.pdb")

inds = []
atomlist = list(pdb.topology.atoms)
for ind, a in enumerate(atomlist):
    if re.search(r"CA$", str(a)):
        inds.append([ind])

cmap = LinearMap(inds, n_fg_sites=coords.shape[1])
constraints = guess_pairwise_constraints(coords[0:10], threshold=1e-3)

# here we deviate from the previous procedure by defining our features
config_feater = Curry(
    gb_feat, inner=0.0, outer=8.0, width=1.0, n_basis=7, batch_size=1000, lazy=True
)
# We combine our feater with id_feat, which assigns a one-hot id to 
feater = Multifeaturize([p.id_feat, config_feater])
optim_results = project_forces(
    xyz=coords,
    forces=forces,
    config_mapping=cmap,
    constrained_inds=constraints,
    l2_regularization=1e3,
    kbt=0.6955215,
    featurizer=feater,
    method=qp_feat_linear_map,
)

# look at examples directory for more details
```

### Testing

Tests are provided via `pytest`, and may be run if installation is performed with the 
`[test]` target. To avoid tests which require `jax`, exclude test with the `jax` marker. 
Note that certain tests use a different quadratic programming back end, `scs`, than 
is default for the main code base.

## License

Copyright 2024 Aleksander Evren Paetzold Durumeric

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
