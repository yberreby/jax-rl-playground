#!/usr/bin/env python3
"""Unit tests for baseline computation."""

import jax.numpy as jnp
from src.baseline import BaselineState, update_baseline, compute_advantages


def test_baseline_tracks_mean_returns():
    """Test that baseline correctly tracks mean of returns."""
    # Start with empty baseline
    baseline = BaselineState(mean=jnp.array(0.0), n_samples=0)
    
    # First batch of returns (episode cumulative returns)
    returns1 = jnp.array([10.0, 20.0, 30.0, 40.0])  # 4 episodes
    baseline = update_baseline(baseline, returns1)
    
    print("After batch 1:")
    print(f"  Returns: {returns1}")
    print(f"  Expected mean: {jnp.mean(returns1)}")
    print(f"  Baseline mean: {baseline.mean}")
    print(f"  Match: {jnp.allclose(baseline.mean, jnp.mean(returns1))}")
    
    # Second batch
    returns2 = jnp.array([5.0, 15.0, 25.0, 35.0])
    baseline = update_baseline(baseline, returns2)
    
    all_returns = jnp.concatenate([returns1, returns2])
    print("\nAfter batch 2:")
    print(f"  All returns: {all_returns}")
    print(f"  Expected mean: {jnp.mean(all_returns)}")
    print(f"  Baseline mean: {baseline.mean}")
    print(f"  Match: {jnp.allclose(baseline.mean, jnp.mean(all_returns))}")
    
    # Check sample count
    print(f"\nSample count: {baseline.n_samples} (expected: {len(all_returns)})")


def test_pendulum_returns_scale():
    """Test what scale pendulum returns are at."""
    # Pendulum rewards are cos(theta), range [-1, 1]
    # Episode length is 200 steps
    
    print("\n=== Pendulum Returns Scale ===")
    
    # Best case: always at top (cos(0) = 1)
    best_rewards = jnp.ones(200)
    best_return = jnp.sum(best_rewards)
    print(f"Best episode return (sum): {best_return}")
    print(f"Best episode return (avg): {best_return / 200}")
    
    # Worst case: always at bottom (cos(π) = -1)
    worst_rewards = -jnp.ones(200)
    worst_return = jnp.sum(worst_rewards)
    print(f"Worst episode return (sum): {worst_return}")
    print(f"Worst episode return (avg): {worst_return / 200}")
    
    # Random case: average cos over uniform angles
    print("\nRandom policy (expected):")
    print("  E[cos(θ)] for θ ~ Uniform[-π, π] = 0")
    print("  Episode return (sum): ~0")
    print("  Episode return (avg): ~0")
    
    # Typical learning curve
    print("\nTypical untrained policy:")
    print("  If average reward is 0.1 per step")
    print(f"  Episode return (sum): {0.1 * 200}")
    print("  Episode return (avg): 0.1")


def test_advantage_computation():
    """Test advantage computation with different baselines."""
    print("\n=== Advantage Computation ===")
    
    # Test returns from 4 episodes
    returns = jnp.array([
        20.0,  # Good episode (avg reward 0.1)
        -10.0,  # Bad episode (avg reward -0.05)
        40.0,  # Great episode (avg reward 0.2)
        10.0,  # Okay episode (avg reward 0.05)
    ])
    
    # Test different baseline values
    for baseline_val in [0.0, 15.0, 20.0]:
        advantages = compute_advantages(returns, baseline_val)
        print(f"\nBaseline = {baseline_val}:")
        print(f"  Returns: {returns}")
        print(f"  Advantages: {advantages}")
        print(f"  Mean advantage: {jnp.mean(advantages):.2f}")
        print(f"  Std advantage: {jnp.std(advantages):.2f}")


if __name__ == "__main__":
    test_baseline_tracks_mean_returns()
    test_pendulum_returns_scale()
    test_advantage_computation()