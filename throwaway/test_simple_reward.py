#!/usr/bin/env python3
"""Test if simple reward helps pendulum learn."""

import jax
import time
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env
from src.train import train, collect_episode

# Quick training test with different learning rates
print("=== Testing Simple Reward ===")
lrs = [3e-3, 1e-2, 3e-2, 5e-2]

for lr in lrs:
    policy = GaussianPolicy(2, 1, hidden_dim=128)
    
    start = time.time()
    metrics = train(
        policy, step, reset_env,
        n_iterations=50,
        episodes_per_iter=16,
        learning_rate=lr,
        use_baseline=True,
        verbose=False,
    )
    train_time = time.time() - start
    
    # Check final performance
    returns = metrics["mean_return"]
    initial = returns[0]
    final = returns[-1]
    best = max(returns)
    
    # Check if actions exploded
    key = jax.random.PRNGKey(123)
    episode = collect_episode(policy, step, reset_env, key)
    max_action = float(jax.numpy.max(jax.numpy.abs(episode.actions)))
    
    print(f"\nlr={lr:.3f}:")
    print(f"  Returns: {initial:.1f} -> {final:.1f} (best: {best:.1f})")
    print(f"  Max action: {max_action:.1f}")
    print(f"  Time: {train_time:.1f}s ({train_time/50:.2f}s per iter)")
    
    # Show learning curve sample
    curve = [returns[i] for i in range(0, len(returns), 10)]
    print(f"  Curve: {[f'{r:.0f}' for r in curve]}")

print("\nNote: With cos(theta) reward, max possible return is 200 (top for 200 steps)")
print("Random policy should get around -120 to -140")