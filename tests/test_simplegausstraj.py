r"""Tests Gaussian Trajectory augmenter `SimpleCondNormal` against the more general but
slower `JCondNormal`.
"""

from typing import Tuple, Final

import numpy as np
import pytest

# this seeds some portions of the randomness of these tests, but not be
# complete.
rseed: Final = 42100


@pytest.mark.jax
def test_simple_cond_normal(seed: int = rseed) -> None:
    from aggforce.trajectory.jaxgausstraj import JCondNormal
    from aggforce.trajectory.simplegausstraj import SimpleCondNormal

    jaugmenter = JCondNormal(0.3, seed=seed)
    saugmenter = SimpleCondNormal.from_jcondnormal(jaugmenter)
    coords = np.zeros((10, 5, 3))
    generated = jaugmenter.sample(coords)
    jlog_grad = jaugmenter.log_gradient(coords, generated)
    slog_grad = saugmenter.log_gradient(coords, generated)
    assert all(
        np.allclose(jarr, sarr, rtol=0, atol=2e-6)
        for jarr, sarr in zip(jlog_grad, slog_grad)
    )
