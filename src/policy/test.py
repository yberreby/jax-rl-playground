import jax
import jax.numpy as jnp
from src.policy import sample_actions
from tests.fixtures import create_test_policy
from tests.constants import DEFAULT_OBS_DIM, DEFAULT_ACTION_DIM, DEFAULT_SEED

# Test dimensions
TEST_HIDDEN_DIM = 32
TEST_BATCH_SIZE = 8


def test_policy_creation():
    policy = create_test_policy(
        obs_dim=DEFAULT_OBS_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        hidden_dim=TEST_HIDDEN_DIM,
    )
    assert policy.w1.value.shape == (DEFAULT_OBS_DIM, TEST_HIDDEN_DIM)
    assert policy.w2.value.shape == (TEST_HIDDEN_DIM, DEFAULT_ACTION_DIM)
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
