#!/usr/bin/env python3
"""Test the episode baseline implementation."""

import jax.numpy as jnp
from src.baseline.episode_baseline import update_episode_baseline, compute_episode_advantages, init_episode_baseline


def test_episode_baseline_updates():
    """Test that episode baseline correctly tracks mean episode returns."""
    print("=== Episode Baseline Updates ===")
    
    baseline = init_episode_baseline()
    print(f"Initial baseline: mean={baseline.mean_episode_return}, n={baseline.n_episodes}")
    
    # First batch of episodes
    episode_totals_1 = jnp.array([20.0, 15.0, 25.0])
    baseline = update_episode_baseline(baseline, episode_totals_1)
    
    print(f"\nAfter batch 1: {episode_totals_1}")
    print(f"Baseline: mean={baseline.mean_episode_return:.2f}, n={baseline.n_episodes}")
    print(f"Expected mean: {jnp.mean(episode_totals_1):.2f}")
    print(f"Match: {jnp.allclose(baseline.mean_episode_return, jnp.mean(episode_totals_1))}")
    
    # Second batch
    episode_totals_2 = jnp.array([10.0, 30.0])
    baseline = update_episode_baseline(baseline, episode_totals_2)
    
    all_episodes = jnp.concatenate([episode_totals_1, episode_totals_2])
    print(f"\nAfter batch 2: {episode_totals_2}")
    print(f"All episodes: {all_episodes}")
    print(f"Baseline: mean={baseline.mean_episode_return:.2f}, n={baseline.n_episodes}")
    print(f"Expected mean: {jnp.mean(all_episodes):.2f}")
    print(f"Match: {jnp.allclose(baseline.mean_episode_return, jnp.mean(all_episodes))}")


def test_episode_advantages():
    """Test episode advantage computation."""
    print("\n=== Episode Advantages ===")
    
    episode_totals = jnp.array([20.0, 15.0, 25.0, 10.0])
    baseline_mean = jnp.array(17.5)
    
    advantages = compute_episode_advantages(episode_totals, baseline_mean)
    
    print(f"Episode totals: {episode_totals}")
    print(f"Baseline mean: {baseline_mean}")
    print(f"Advantages: {advantages}")
    print(f"Expected: {episode_totals - baseline_mean}")
    print(f"Match: {jnp.allclose(advantages, episode_totals - baseline_mean)}")
    
    # Check that mean advantage is zero (when baseline is accurate)
    baseline_mean_exact = jnp.mean(episode_totals)
    advantages_exact = compute_episode_advantages(episode_totals, baseline_mean_exact)
    print(f"\nWith exact baseline ({baseline_mean_exact:.1f}):")
    print(f"Advantages: {advantages_exact}")
    print(f"Mean advantage: {jnp.mean(advantages_exact):.6f} (should be ~0)")


def test_display_conversion():
    """Test conversion to display values."""
    print("\n=== Display Conversion ===")
    
    baseline_mean = jnp.array(20.0)  # Mean episode total
    episode_length = 200
    
    display_value = baseline_mean / episode_length
    
    print(f"Baseline (mean episode total): {baseline_mean}")
    print(f"Episode length: {episode_length}")
    print(f"Display value (avg reward/step): {display_value}")
    print("This is clean and interpretable!")


if __name__ == "__main__":
    test_episode_baseline_updates()
    test_episode_advantages() 
    test_display_conversion()