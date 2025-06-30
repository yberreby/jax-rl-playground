#!/usr/bin/env python3
"""Test baseline behavior in actual training context."""

import jax.numpy as jnp
from src.train import compute_returns


def test_returns_vs_rewards():
    """Test the difference between returns and rewards."""
    print("=== Returns vs Rewards ===")
    
    # Simulate episode rewards (per-step rewards)
    rewards = jnp.array([0.1, 0.2, -0.1, 0.15, 0.05])  # 5 steps
    
    # Compute returns (cumulative from each step)
    returns = compute_returns(rewards)
    
    print(f"Per-step rewards: {rewards}")
    print(f"Returns: {returns}")
    print(f"Total episode reward (sum): {jnp.sum(rewards)}")
    print(f"Average per-step reward: {jnp.mean(rewards)}")
    print(f"First return (total from start): {returns[0]}")
    
    # What baseline should track
    print("\nWhat baseline tracks:")
    print(f"  All returns: {returns}")
    print(f"  Mean return: {jnp.mean(returns)}")
    
    print("\nWhat we display as 'mean_return':")
    print(f"  Average per-step reward: {jnp.mean(rewards)}")
    
    print(f"\nMismatch factor: {jnp.mean(returns) / jnp.mean(rewards):.1f}x")


def test_pendulum_episode_structure():
    """Test what happens with typical pendulum episode structure."""
    print("\n=== Pendulum Episode Structure ===")
    
    # Typical pendulum episode: 200 steps, rewards around 0.1
    episode_length = 200
    avg_reward = 0.1
    rewards = jnp.full(episode_length, avg_reward)
    
    returns = compute_returns(rewards)
    
    print(f"Episode length: {episode_length}")
    print(f"Average reward per step: {avg_reward}")
    print(f"Total episode reward: {jnp.sum(rewards)}")
    print(f"First return (total remaining): {returns[0]}")
    print(f"Last return (single step): {returns[-1]}")
    print(f"Mean return: {jnp.mean(returns)}")
    
    # This is what baseline tracks
    baseline_value = jnp.mean(returns)
    print(f"\nBaseline tracks: {baseline_value}")
    print(f"Display shows: {avg_reward}")
    print(f"Current hack (baseline/200): {baseline_value/200}")
    print(f"Ratio baseline/display: {baseline_value/avg_reward:.1f}x")


def test_correct_baseline_computation():
    """Test what the correct baseline should be."""
    print("\n=== Correct Baseline Computation ===")
    
    # Multiple episodes with different performance
    episodes = [
        jnp.array([0.05] * 200),  # Poor episode
        jnp.array([0.15] * 200),  # Good episode  
        jnp.array([0.10] * 200),  # Average episode
    ]
    
    all_returns = []
    episode_totals = []
    
    for i, rewards in enumerate(episodes):
        returns = compute_returns(rewards)
        all_returns.append(returns)
        episode_total = jnp.sum(rewards)
        episode_totals.append(episode_total)
        
        print(f"Episode {i+1}:")
        print(f"  Avg reward/step: {jnp.mean(rewards)}")
        print(f"  Total reward: {episode_total}")
        print(f"  Mean return: {jnp.mean(returns)}")
    
    # What baseline currently tracks
    all_returns_flat = jnp.concatenate(all_returns)
    current_baseline = jnp.mean(all_returns_flat)
    
    # What we display
    episode_totals = jnp.array(episode_totals)
    display_mean = jnp.mean(episode_totals) / 200
    
    print("\nSummary:")
    print(f"  Baseline tracks: {current_baseline:.2f}")
    print(f"  Display shows: {display_mean:.3f}")
    print(f"  Ratio: {current_baseline/display_mean:.1f}x")
    
    # The issue: baseline tracks mean of ALL returns in episode
    # But we want to compare against episode total rewards
    print("\nCorrect comparison:")
    print(f"  Episode totals: {episode_totals}")
    print(f"  Mean episode total: {jnp.mean(episode_totals):.2f}")
    print(f"  This should match baseline/100.5: {current_baseline/100.5:.2f}")


if __name__ == "__main__":
    test_returns_vs_rewards()
    test_pendulum_episode_structure() 
    test_correct_baseline_computation()