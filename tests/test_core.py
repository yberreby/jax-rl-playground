import jax.numpy as jnp
import numpy as np
import pytest
from src.core import exponential_moving_average, batch_correlate


def test_ema_shape():
    values = jnp.ones((2, 10, 3))
    alpha = jnp.array(0.1)
    result = exponential_moving_average(values, alpha)
    assert result.shape == (2, 10, 3)


def test_ema_with_window_size():
    values = jnp.arange(6).reshape(1, 6, 1).astype(float)
    result = exponential_moving_average(values, jnp.array(0.0), window_size=3)
    assert result.shape == (1, 6, 1)


def test_ema_constant_input():
    values = jnp.ones((1, 5, 1)) * 3.0
    alpha = jnp.array(0.2)
    result = exponential_moving_average(values, alpha)

    np.testing.assert_allclose(result, 3.0, rtol=1e-6)


def test_batch_correlate_shape():
    x = jnp.ones((3, 5), dtype=float)
    y = jnp.ones((3, 5), dtype=float)
    result = batch_correlate(x, y)
    assert result.shape == (3, 9)


def test_batch_correlate_identical():
    x = jnp.array([[1.0, 0.0, 1.0, 0.0, 1.0]])
    result = batch_correlate(x, x)

    expected_peak = jnp.argmax(result[0])
    assert expected_peak == 4  # Should peak at center


def test_batch_correlate_orthogonal():
    x = jnp.array([[1.0, 0.0, 1.0, 0.0]])
    y = jnp.array([[0.0, 1.0, 0.0, 1.0]])
    result = batch_correlate(x, y)

    assert result.shape == (1, 7)
    np.testing.assert_allclose(result[0, 3], 0.0, atol=1e-6)  # Center should be 0


def test_correlation_symmetry():
    x = jnp.array([[1.0, 2.0, 3.0]])
    y = jnp.array([[4.0, 5.0, 6.0]])

    xy = batch_correlate(x, y)
    yx = batch_correlate(y, x)

    # Cross-correlation has corr(x,y) = reverse(corr(y,x))
    np.testing.assert_allclose(xy, jnp.flip(yx, axis=-1), rtol=1e-6)


def test_jaxtyping_shape_validation():
    """Test that jaxtyping catches shape mismatches at runtime."""
    # Test 1: Wrong number of dimensions for exponential_moving_average
    with pytest.raises(Exception):  # jaxtyping will raise a validation error
        values = jnp.ones((10, 3))  # 2D instead of 3D
        alpha = jnp.array(0.1)
        exponential_moving_average(values, alpha)

    # Test 2: Wrong batch dimensions for batch_correlate
    with pytest.raises(Exception):  # jaxtyping will raise a validation error
        x = jnp.ones((3, 5))  # correct shape
        y = jnp.ones((4, 5))  # wrong batch size
        batch_correlate(x, y)
