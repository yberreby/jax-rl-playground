import jax
import jax.numpy as jnp
from src.policy_nnx import GaussianPolicy, sample_actions


def test_policy_creation():
    policy = GaussianPolicy(obs_dim=4, action_dim=2, hidden_dim=32)
    assert policy.w1.value.shape == (4, 32)
    assert policy.w2.value.shape == (32, 2)
    assert policy.log_std.value.shape == (2,)


def test_policy_forward():
    policy = GaussianPolicy(obs_dim=4, action_dim=2)
    obs = jnp.ones((8, 4))
    
    mean, std = policy(obs)
    assert mean.shape == (8, 2)
    assert std.shape == (8, 2)
    assert jnp.all(std > 0)


def test_sample_actions():
    policy = GaussianPolicy(obs_dim=4, action_dim=2)
    obs = jnp.ones((8, 4))
    key = jax.random.PRNGKey(0)
    
    actions, log_probs = sample_actions(policy, obs, key)
    assert actions.shape == (8, 2)
    assert log_probs.shape == (8,)
    assert jnp.all(jnp.isfinite(log_probs))