#!/usr/bin/env python3
"""Test simple episode-level baseline."""

import jax.numpy as jnp


def test_simple_episode_baseline():
    """Test tracking just episode totals."""
    print("=== Simple Episode-Level Baseline ===")

    # Episode total returns (what we actually care about)
    episode_totals = jnp.array([20.0, 15.0, 25.0, 10.0])  # 4 episodes

    baseline = jnp.mean(episode_totals)

    print(f"Episode totals: {episode_totals}")
    print(f"Baseline (mean episode total): {baseline}")
    print(f"Display value: {baseline / 200} (avg reward per step)")

    # Advantages for each episode
    episode_advantages = episode_totals - baseline
    print(f"Episode advantages: {episode_advantages}")

    # This is clean and interpretable!
    print("\nWhat this means:")
    print(
        f"- Episode 1: {episode_totals[0]:.0f} return, {episode_advantages[0]:+.0f} vs baseline"
    )
    print(
        f"- Episode 2: {episode_totals[1]:.0f} return, {episode_advantages[1]:+.0f} vs baseline"
    )
    print(
        f"- Episode 3: {episode_totals[2]:.0f} return, {episode_advantages[2]:+.0f} vs baseline"
    )
    print(
        f"- Episode 4: {episode_totals[3]:.0f} return, {episode_advantages[3]:+.0f} vs baseline"
    )


def test_what_about_timestep_advantages():
    """What if we need timestep-level advantages?"""
    print("\n=== Timestep-Level Advantages ===")

    # If we need advantages at each timestep, we can:
    # Option 1: Use episode-level advantage for all timesteps
    episode_total = 20.0
    baseline = 17.5  # From episode-level baseline
    episode_advantage = episode_total - baseline

    print(f"Episode total: {episode_total}")
    print(f"Episode baseline: {baseline}")
    print(f"Episode advantage: {episode_advantage}")
    print(f"Use same advantage for all 200 timesteps: {episode_advantage}")

    print("\nThis is much cleaner than our current approach!")


if __name__ == "__main__":
    test_simple_episode_baseline()
    test_what_about_timestep_advantages()
