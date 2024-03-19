r"""Test `SimpleCondNormal` against `JCondNormal`."""

from typing import Final, Tuple

import numpy as np
import pytest

# this seeds some portions of the randomness of these tests, but not be
# complete.
rseed: Final = 42100


@pytest.mark.jax
def test_simple_cond_normal(seed: int = rseed) -> None:
    """Test to see if log gradients match for JCondNormal and SimpleCondNormal."""
    NOISE_LEVEL: Final = 0.3
    COORD_SHAPE: Tuple = (10, 5, 3)
    from aggforce.trajectory.jaxgausstraj import JCondNormal

    jaugmenter = JCondNormal(NOISE_LEVEL, seed=seed)
    saugmenter = jaugmenter.to_SimpleCondNormal()
    coords = np.zeros(COORD_SHAPE)
    generated = jaugmenter.sample(coords)
    jlog_grad = jaugmenter.log_gradient(coords, generated)
    slog_grad = saugmenter.log_gradient(coords, generated)
    assert all(
        np.allclose(jarr, sarr, rtol=0, atol=2e-6)
        for jarr, sarr in zip(jlog_grad, slog_grad)
    )
