import jax
import jax.numpy as jnp
import numpy as np
from src.policy import policy_forward, init_policy, sample_actions


def test_policy_forward_shapes():
    key = jax.random.PRNGKey(0)

    # Single test case to avoid multiple JIT compilations
    params = init_policy(key, obs_dim=8, action_dim=2)

    for batch_size in [1, 16, 32]:
        obs = jax.random.normal(key, (batch_size, 8))
        mean, std = policy_forward(params, obs)

        assert mean.shape == (batch_size, 2)
        assert std.shape == (batch_size, 2)
        assert jnp.all(std > 0)


def test_sample_actions_distribution():
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros((1000, 2))
    std = jnp.ones((1000, 2))

    actions, log_probs = sample_actions(key, mean, std)

    empirical_mean = jnp.mean(actions, axis=0)
    empirical_std = jnp.std(actions, axis=0)

    np.testing.assert_allclose(empirical_mean, 0.0, atol=0.1)
    np.testing.assert_allclose(empirical_std, 1.0, atol=0.1)


def test_log_probs_correct():
    key = jax.random.PRNGKey(0)
    mean = jnp.ones((5, 3))
    std = jnp.ones((5, 3)) * 0.5

    actions, log_probs = sample_actions(key, mean, std)

    manual_log_probs = -0.5 * jnp.sum(
        jnp.square((actions - mean) / std) + 2 * jnp.log(std) + jnp.log(2 * jnp.pi),
        axis=-1,
    )

    np.testing.assert_allclose(log_probs, manual_log_probs)
