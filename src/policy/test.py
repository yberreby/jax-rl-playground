import jax
import jax.numpy as jnp
from flax import nnx
from src.policy import sample_actions, GaussianPolicy
from src.constants import DEFAULT_OBS_DIM, DEFAULT_ACTION_DIM, DEFAULT_SEED

# Test dimensions
TEST_HIDDEN_DIM = 32
TEST_BATCH_SIZE = 8


def create_test_policy(
    obs_dim: int = DEFAULT_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    hidden_dim: int = TEST_HIDDEN_DIM,
) -> GaussianPolicy:
    rngs = nnx.Rngs(DEFAULT_SEED)
    return GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        rngs=rngs,
    )


def test_policy_creation():
    policy = create_test_policy(
        obs_dim=DEFAULT_OBS_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        hidden_dim=TEST_HIDDEN_DIM,
    )
    # Policy now uses layers list with feature encoding
    assert len(policy.layers) > 0
    assert policy.log_std.value.shape == (DEFAULT_ACTION_DIM,)


def test_policy_forward():
    policy = create_test_policy()
    obs = jnp.ones((TEST_BATCH_SIZE, DEFAULT_OBS_DIM))

    mean, std = policy(obs)
    assert mean.shape == (TEST_BATCH_SIZE, DEFAULT_ACTION_DIM)
    assert std.shape == (TEST_BATCH_SIZE, DEFAULT_ACTION_DIM)
    assert jnp.all(std > 0)


def test_sample_actions():
    policy = create_test_policy()
    obs = jnp.ones((TEST_BATCH_SIZE, DEFAULT_OBS_DIM))
    key = jax.random.PRNGKey(DEFAULT_SEED)

    actions, log_probs = sample_actions(policy, obs, key)
    assert actions.shape == (TEST_BATCH_SIZE, DEFAULT_ACTION_DIM)
    assert log_probs.shape == (TEST_BATCH_SIZE,)
    assert jnp.all(jnp.isfinite(log_probs))
