import jax.numpy as jnp
import numpy as np
from src.distributions import gaussian_log_prob


def test_gaussian_log_prob_standard_normal():
    # Test against standard normal at x=0
    x = jnp.zeros(1)
    mean = jnp.zeros(1)
    std = jnp.ones(1)
    
    log_prob = gaussian_log_prob(x, mean, std)
    expected = -0.5 * jnp.log(2 * jnp.pi)
    
    np.testing.assert_allclose(log_prob, expected, rtol=1e-6)


def test_gaussian_log_prob_batch():
    x = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    mean = jnp.zeros((2, 2))
    std = jnp.ones((2, 2))
    
    log_probs = gaussian_log_prob(x, mean, std)
    assert log_probs.shape == (2,)


def test_gaussian_log_prob_matches_scipy():
    # Compare with manual calculation
    x = jnp.array([1.5])
    mean = jnp.array([0.5])
    std = jnp.array([2.0])
    
    z = (x - mean) / std
    expected = -0.5 * (z**2 + jnp.log(2 * jnp.pi) + 2 * jnp.log(std))
    actual = gaussian_log_prob(x, mean, std)
    
    np.testing.assert_allclose(actual, expected, rtol=1e-6)