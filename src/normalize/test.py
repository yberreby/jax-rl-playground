import jax
import jax.numpy as jnp
import numpy as np
from src.normalize import normalize_obs, scale_rewards


def test_normalize_obs():
    obs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    norm = normalize_obs(obs)
    assert norm.shape == obs.shape
    assert jnp.all(jnp.abs(norm) <= 5.0)


def test_scale_rewards():
    rewards = jnp.ones(10)
    returns = jnp.arange(10.0)
    scaled = scale_rewards(rewards, returns)
    assert scaled.shape == rewards.shape


def test_normalize_preserves_shape():
    shapes = [(10,), (5, 8), (3, 4, 6)]
    for shape in shapes:
        x = jax.random.normal(jax.random.PRNGKey(0), shape)
        norm_x = normalize_obs(x)
        assert norm_x.shape == x.shape


def test_normalize_stats():
    x = jax.random.normal(jax.random.PRNGKey(0), (100, 8)) * 10 + 5
    norm_x = normalize_obs(x)

    mean = jnp.mean(norm_x, axis=-1)
    std = jnp.std(norm_x, axis=-1)

    np.testing.assert_allclose(mean, 0.0, atol=1e-6)
    np.testing.assert_allclose(std, 1.0, atol=1e-6)


def test_scale_rewards_preserves_sign():
    rewards = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    returns = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    scaled = scale_rewards(rewards, returns)
    assert jnp.all(jnp.sign(rewards) == jnp.sign(scaled))


def test_scale_rewards_zero_variance():
    rewards = jnp.ones(10)
    returns = jnp.ones(10)

    scaled = scale_rewards(rewards, returns)
    assert jnp.all(jnp.isfinite(scaled))
