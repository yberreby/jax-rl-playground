import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.init import sparse_init

# Test constants
LARGE_MATRIX_SIZE = 1000
SPARSITY_TOLERANCE = 0.05
TEST_SPARSITIES = [0.5, 0.7, 0.9]
TEST_FAN_INS = [10, 100, 1000]
SCALE_TOLERANCE_FACTOR = 0.1


def test_sparse_init_sparsity():
    key = jax.random.PRNGKey(42)

    for sparsity in TEST_SPARSITIES:
        W = sparse_init(key, (LARGE_MATRIX_SIZE, LARGE_MATRIX_SIZE), sparsity)
        actual_sparsity = jnp.mean(W == 0)
        assert abs(actual_sparsity - sparsity) < SPARSITY_TOLERANCE


@pytest.mark.slow
def test_sparse_init_scaling():
    key = jax.random.PRNGKey(42)
    n_rows = 100
    test_sparsity = 0.5

    for fan_in in TEST_FAN_INS:
        W = sparse_init(key, (n_rows, fan_in), sparsity=test_sparsity)
        nonzero_values = W[W != 0]
        expected_scale = 1.0 / np.sqrt(fan_in)
        actual_scale = jnp.std(nonzero_values)
        assert (
            abs(actual_scale - expected_scale) < SCALE_TOLERANCE_FACTOR * expected_scale
        )


def test_sparse_init_shapes():
    key = jax.random.PRNGKey(0)
    shapes = [(10, 20), (5,), (3, 4, 5)]

    for shape in shapes:
        W = sparse_init(key, shape)
        assert W.shape == shape
