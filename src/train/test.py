import jax
import jax.numpy as jnp
import pytest
from flax import nnx
import optax
from . import compute_returns, collect_episode, train_step, train
from ..policy import GaussianPolicy
from ..pendulum import step, reset_env

# Test constants
TEST_EPISODE_LENGTH = 10
TEST_BATCH_SIZE = 5
TEST_OBS_DIM = 2
TEST_ACTION_DIM = 1
TEST_HIDDEN_DIM = 16
RETURN_TOLERANCE = 1e-5


def test_compute_returns():
    # Test undiscounted returns computation
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
    expected = jnp.array([10.0, 9.0, 7.0, 4.0])  # Cumulative from each step

    returns = compute_returns(rewards)
    assert jnp.allclose(returns, expected, atol=RETURN_TOLERANCE)


def test_compute_returns_single_reward():
    rewards = jnp.array([5.0])
    returns = compute_returns(rewards)
    assert jnp.allclose(returns, jnp.array([5.0]))


def test_collect_episode_shapes():
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(
        obs_dim=TEST_OBS_DIM,
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
    assert episode.total_reward.shape == ()


def test_train_step_gradient_flow():
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(
        obs_dim=TEST_OBS_DIM, action_dim=TEST_ACTION_DIM, hidden_dim=TEST_HIDDEN_DIM
    )
    optimizer = nnx.Optimizer(policy, optax.adam(1e-3))

    # Create dummy batch
    batch_states = jax.random.normal(key, (TEST_BATCH_SIZE, TEST_OBS_DIM))
    batch_actions = jax.random.normal(key, (TEST_BATCH_SIZE, TEST_ACTION_DIM))
    batch_advantages = jnp.ones(TEST_BATCH_SIZE)

    # Store initial params - get first layer's params
    initial_params = nnx.state(policy.layers[0])

    # Train step
    loss, grad_norm, grad_var = train_step(policy, optimizer, batch_states, batch_actions, batch_advantages)

    # Check gradient flow
    assert loss.shape == ()
    assert grad_norm > 0, "No gradients computed"
    # Check that params updated
    new_params = nnx.state(policy.layers[0])
    params_changed = False
    for k in initial_params:
        if not jnp.array_equal(initial_params[k].value, new_params[k].value):
            params_changed = True
            break
    assert params_changed, "Parameters didn't update"


@pytest.mark.slow
def test_train_improves_performance():
    # Test that training actually improves returns
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(
        obs_dim=2,  # Pendulum state dim
        action_dim=1,  # Pendulum action dim
        hidden_dim=32,
    )

    # Collect initial performance
    initial_episodes = [collect_episode(policy, step, reset_env, key) for _ in range(5)]
    initial_returns = jnp.mean(jnp.array([ep.total_reward for ep in initial_episodes]))

    # Train
    metrics = train(
        policy,
        step,
        reset_env,
        n_iterations=50,
        episodes_per_iter=5,
        learning_rate=3e-3,
        use_baseline=True,
        verbose=False,
    )

    # Check improvement
    final_return = metrics["mean_return"][-1]
    assert final_return > initial_returns, (
        f"No improvement: {initial_returns} -> {final_return}"
    )

    # Check metrics structure
    assert len(metrics["iteration"]) == 50
    assert all(
        key in metrics
        for key in ["mean_return", "std_return", "loss", "baseline_value"]
    )
