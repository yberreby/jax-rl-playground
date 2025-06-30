import jax
import jax.numpy as jnp
from . import compute_returns, collect_episode, EpisodeResult
from ...policy import GaussianPolicy
from ...pendulum import step, reset_env

# Test constants
TEST_EPISODE_LENGTH = 10
TEST_OBS_DIM = 2
TEST_ACTION_DIM = 1
TEST_HIDDEN_DIM = 16
RETURN_TOLERANCE = 1e-5


def test_compute_returns():
    # Test undiscounted returns computation
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
    expected = jnp.array([10.0, 9.0, 7.0, 4.0])  # Cumulative from each step

    returns = compute_returns(rewards, gamma=1.0)
    assert jnp.allclose(returns, expected, atol=RETURN_TOLERANCE)


def test_compute_returns_single_reward():
    rewards = jnp.array([5.0])
    returns = compute_returns(rewards, gamma=1.0)
    assert jnp.allclose(returns, jnp.array([5.0]))


def test_collect_episode_shapes():
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(
        obs_dim=8,  # Features dimension
        action_dim=TEST_ACTION_DIM,
        hidden_dim=TEST_HIDDEN_DIM,
        use_layernorm=False,
    )

    episode = collect_episode(
        policy, step, reset_env, key, max_steps=TEST_EPISODE_LENGTH
    )

    # Check shapes
    assert episode.states.shape[1] == TEST_OBS_DIM
    assert episode.actions.shape[1] == TEST_ACTION_DIM
    assert episode.rewards.shape == episode.returns.shape
    assert episode.log_probs.shape == episode.rewards.shape
    assert episode.total_reward.shape == ()


def test_episode_result_structure():
    # Test that EpisodeResult can be created correctly
    states = jnp.ones((10, 2))
    actions = jnp.ones((10, 1))
    rewards = jnp.ones(10)
    returns = jnp.ones(10) * 10
    total_reward = jnp.array(1.0)
    log_probs = jnp.ones(10) * -0.5
    
    result = EpisodeResult(
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        total_reward=total_reward,
        log_probs=log_probs,
    )
    
    assert result.states.shape == (10, 2)
    assert result.total_reward == 1.0