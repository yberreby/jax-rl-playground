#!/usr/bin/env python3
"""Train pendulum with fixed bounded actions and proper rewards."""

import jax
import time
import numpy as np
import matplotlib.pyplot as plt
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env
from src.train import train, collect_episode
from src.viz.pendulum import PendulumVisualizer

# Train with reasonable hyperparameters
print("=== Training Pendulum Swing-Up ===")
print("Reward: cos(theta), range [-1, 1], avg per step")
print("Actions: bounded to [-2, 2] via tanh")

policy = GaussianPolicy(
    obs_dim=2, 
    action_dim=1, 
    hidden_dim=128,
    use_layernorm=True
)

start = time.time()
metrics = train(
    policy,
    step,
    reset_env,
    n_iterations=200,
    episodes_per_iter=32,
    learning_rate=1e-2,
    use_baseline=True,
    seed=42,
    verbose=True,
)
train_time = time.time() - start

print(f"\nTraining completed in {train_time:.1f}s")

# Analyze final performance
returns = np.array(metrics["mean_return"])
print("\nPerformance:")
print(f"  Initial: {returns[0]:.3f}")
print(f"  Final: {returns[-1]:.3f}")
print(f"  Best: {returns.max():.3f}")
print(f"  Improvement: {returns[-1] - returns[0]:.3f}")

# Evaluate and visualize
print("\n=== Evaluating Trained Policy ===")
key = jax.random.PRNGKey(123)
episode = collect_episode(policy, step, reset_env, key, max_steps=200)

print(f"Episode reward: {episode.total_reward:.3f}")

# Check if it reaches the top
states = episode.states
angles = states[:, 0]
heights = jax.numpy.cos(angles)  # 1 = top, -1 = bottom
max_height = float(heights.max())
time_at_top = float(jax.numpy.sum(heights > 0.9))

print(f"Max height achieved: {max_height:.3f} (1.0 = top)")
print(f"Steps near top (cos > 0.9): {time_at_top:.0f}/200")

# Create visualization
try:
    viz = PendulumVisualizer()
    animation = viz.create_animation(
        [s for s in episode.states],
        [a for a in episode.actions], 
        [r for r in episode.rewards],
    )
    animation.save("tests/outputs/pendulum_final.mp4", writer="ffmpeg", fps=20)
    print("\nSaved animation to tests/outputs/pendulum_final.mp4")
except Exception as e:
    print(f"\nAnimation failed: {e}")

# Save learning curve
plt.figure(figsize=(10, 6))
plt.plot(metrics["iteration"], returns)
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.title("Pendulum Learning Curve")
plt.grid(True, alpha=0.3)
plt.savefig("tests/outputs/pendulum_learning_curve.png")
print("Saved learning curve to tests/outputs/pendulum_learning_curve.png")