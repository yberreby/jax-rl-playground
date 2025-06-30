#!/usr/bin/env python3
"""Pendulum swing-up tests."""

import pytest
import jax
import jax.numpy as jnp
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env, MAX_EPISODE_STEPS
from src.train import collect_episodes


def test_pendulum_environment():
    """Test basic pendulum environment functionality."""
    key = jax.random.PRNGKey(0)
    
    # Test reset
    env_state = reset_env(key)
    assert env_state.state.shape == (2,)
    assert env_state.step_count == 0
    
    # Test step
    action = jnp.array([1.0])
    result = step(env_state, action)
    assert result.state.shape == (2,)
    assert result.reward.shape == ()
    assert result.env_state.step_count == 1
    
    # Test episode termination
    for _ in range(MAX_EPISODE_STEPS - 1):
        result = step(result.env_state, action)
    assert result.done == 1.0


def test_policy_with_features():
    """Test that policy uses features correctly."""
    policy = GaussianPolicy(obs_dim=2, action_dim=1, hidden_dim=64)
    
    # Test forward pass
    obs = jnp.array([[0.0, 0.0], [1.0, 0.5]])
    mean, std = policy(obs)
    
    assert mean.shape == (2, 1)
    assert std.shape == (2, 1)
    assert jnp.all(std > 0)


def test_episode_collection():
    """Test episode collection with proper length."""
    policy = GaussianPolicy(obs_dim=2, action_dim=1, hidden_dim=32)
    key = jax.random.PRNGKey(0)
    
    # Collect episodes
    episode_batch = collect_episodes(
        policy, step, reset_env, key, n_episodes=2
    )
    
    # Check shapes
    assert episode_batch.states.shape == (2 * MAX_EPISODE_STEPS, 2)
    assert episode_batch.actions.shape == (2 * MAX_EPISODE_STEPS, 1)
    assert episode_batch.rewards.shape == (2 * MAX_EPISODE_STEPS,)
    
    # Check that rewards are non-zero (pendulum always has cost)
    assert jnp.sum(episode_batch.rewards != 0.0) == 2 * MAX_EPISODE_STEPS


@pytest.mark.slow
def test_short_training_run():
    """Test that training runs without errors."""
    from src.train import train
    
    policy = GaussianPolicy(obs_dim=2, action_dim=1, hidden_dim=64)
    
    metrics = train(
        policy,
        step,
        reset_env,
        n_iterations=10,
        episodes_per_iter=10,
        learning_rate=1e-4,
        use_critic=True,
        verbose=False
    )
    
    assert len(metrics['iteration']) == 10
    assert all(r < 0 for r in metrics['mean_return'])  # Pendulum starts bad