#!/usr/bin/env python3
"""Pendulum swing-up tests."""

import pytest
import jax
import jax.numpy as jnp
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env, MAX_EPISODE_STEPS
from src.pendulum.features import compute_features
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


def test_pendulum_features():
    """Test that pendulum features are computed correctly."""
    # Test feature computation
    raw_obs = jnp.array([[0.0, 0.0], [jnp.pi, 1.0]])  # down and up positions
    features = jax.vmap(compute_features)(raw_obs)
    
    assert features.shape == (2, 8)
    
    # Check specific feature values for down position (theta=0)
    down_features = features[0]
    assert jnp.allclose(down_features[0], 0.0, atol=1e-6)  # sin(0) = 0
    assert jnp.allclose(down_features[1], 1.0, atol=1e-6)  # cos(0) = 1
    
    # Check specific feature values for up position (theta=pi)
    up_features = features[1]
    assert jnp.allclose(up_features[0], 0.0, atol=1e-6)  # sin(pi) ≈ 0
    assert jnp.allclose(up_features[1], -1.0, atol=1e-6)  # cos(pi) = -1


def test_episode_collection():
    """Test episode collection with proper length."""
    # Policy expects 8D features
    policy = GaussianPolicy(obs_dim=8, action_dim=1, hidden_dim=32)
    key = jax.random.PRNGKey(0)
    
    # Collect episodes
    episode_batch = collect_episodes(
        policy, step, reset_env, key, n_episodes=2
    )
    
    # Check shapes - states are raw 2D pendulum states
    assert episode_batch.states.shape == (2 * MAX_EPISODE_STEPS, 2)
    assert episode_batch.actions.shape == (2 * MAX_EPISODE_STEPS, 1)
    assert episode_batch.rewards.shape == (2 * MAX_EPISODE_STEPS,)
    
    # Check that rewards are non-zero (pendulum always has cost)
    assert jnp.sum(episode_batch.rewards != 0.0) == 2 * MAX_EPISODE_STEPS


@pytest.mark.slow
def test_quick_episode_visualization():
    """Quick test that generates episode videos for feedback."""
    import time
    from pathlib import Path
    from src.train.episodes import collect_episode
    from src.viz.pendulum import PendulumVisualizer
    
    # Create output directory
    output_dir = Path("tests/outputs/pendulum_quick")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    policy = GaussianPolicy(obs_dim=8, action_dim=1, hidden_dim=64)
    key = jax.random.PRNGKey(42)
    viz = PendulumVisualizer()
    
    print("\n=== Quick Pendulum Episode Test ===")
    
    # Collect a few episodes with timing
    for i in range(3):
        start_time = time.time()
        key, subkey = jax.random.split(key)
        episode = collect_episode(policy, step, reset_env, subkey, max_steps=MAX_EPISODE_STEPS)
        collect_time = time.time() - start_time
        
        # Convert to lists for visualization (states are raw 2D)
        states = [episode.states[t] for t in range(MAX_EPISODE_STEPS)]
        actions = [episode.actions[t] for t in range(MAX_EPISODE_STEPS)]
        rewards = [episode.rewards[t] for t in range(MAX_EPISODE_STEPS)]
        
        # Generate video
        start_time = time.time()
        viz.create_animation(states, actions, rewards, str(output_dir / f"episode_{i}.mp4"))
        viz_time = time.time() - start_time
        
        print(f"Episode {i}: return={episode.total_reward:.1f}, "
              f"collect_time={collect_time:.3f}s, viz_time={viz_time:.3f}s")
    
    # Check outputs exist
    assert (output_dir / "episode_0.mp4").exists()
    assert (output_dir / "episode_1.mp4").exists()
    assert (output_dir / "episode_2.mp4").exists()
    
    print(f"\nVideos saved to: {output_dir}")
    

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