#!/usr/bin/env python3
"""Test the correct way to display baseline vs returns."""

import jax.numpy as jnp
from src.train import compute_returns


def test_baseline_display_formula():
    """Test the mathematical relationship between baseline and display."""
    print("=== Baseline Display Formula ===")

    episode_lengths = [50, 100, 200, 1000]

    for length in episode_lengths:
        avg_reward = 0.1
        rewards = jnp.full(length, avg_reward)
        returns = compute_returns(rewards)
        mean_return = jnp.mean(returns)

        # Mathematical formula: mean return = avg_reward * (length + 1) / 2
        # This is the sum of arithmetic series divided by n
        expected_mean_return = avg_reward * (length + 1) / 2

        print(f"\nEpisode length: {length}")
        print(f"  Avg reward/step: {avg_reward}")
        print(f"  Mean return (computed): {mean_return:.3f}")
        print(f"  Mean return (formula): {expected_mean_return:.3f}")
        print(f"  Match: {jnp.allclose(mean_return, expected_mean_return)}")
        print(f"  Conversion factor: {mean_return / avg_reward:.1f}")
        print(f"  Formula factor: {(length + 1) / 2:.1f}")


def test_correct_baseline_conversion():
    """Test correct conversion from baseline to display value."""
    print("\n=== Correct Baseline Conversion ===")

    episode_length = 200
    baseline_value = 10.05  # From our test

    # Correct conversion: baseline = avg_reward * (length + 1) / 2
    # So: avg_reward = baseline * 2 / (length + 1)
    display_value = baseline_value * 2 / (episode_length + 1)

    print(f"Baseline value: {baseline_value}")
    print(f"Episode length: {episode_length}")
    print(f"Correct display value: {display_value:.3f}")
    print(f"Current hack (divide by 200): {baseline_value / 200:.3f}")
    print(f"Difference: {abs(display_value - baseline_value / 200):.3f}")


if __name__ == "__main__":
    test_baseline_display_formula()
    test_correct_baseline_conversion()
