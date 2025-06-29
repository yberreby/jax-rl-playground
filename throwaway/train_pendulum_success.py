#!/usr/bin/env python3
"""Train pendulum to successfully swing up with proper tracking."""

import jax
import time
import numpy as np
import matplotlib.pyplot as plt
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env
from src.train import train, collect_episode

print("=== Training Pendulum Swing-Up ===")
print("Goal: Swing up and balance at the top")
print("Success metric: >150 steps at top (cos(θ) > 0.9) out of 200")

# Train with good hyperparameters
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
    n_iterations=300,  # Longer training
    episodes_per_iter=64,  # More episodes for stability
    learning_rate=5e-3,  # Conservative learning rate
    use_baseline=True,
    seed=42,
    verbose=True,
)
train_time = time.time() - start

print(f"\nTraining completed in {train_time:.1f}s")
print(f"Time per iteration: {train_time/300:.2f}s")

# Evaluate final performance
print("\n=== Final Evaluation ===")
key = jax.random.PRNGKey(456)
successes = 0
total_steps_at_top = 0
episode_rewards = []

for i in range(50):
    key, ep_key = jax.random.split(key)
    episode = collect_episode(policy, step, reset_env, ep_key)
    
    # Track reward
    episode_rewards.append(float(episode.total_reward))
    
    # Count time at top
    heights = jax.numpy.cos(episode.states[:, 0])
    steps_at_top = float(jax.numpy.sum(heights > 0.9))
    total_steps_at_top += steps_at_top
    
    if steps_at_top > 150:  # More than 75% of episode
        successes += 1
    
    if i < 5:  # Print first few episodes
        print(f"Episode {i+1}: reward={episode.total_reward:.3f}, steps at top={steps_at_top:.0f}/200")

print(f"\nSuccess rate (>150 steps at top): {successes}/50 = {successes/50:.0%}")
print(f"Average steps at top: {total_steps_at_top/50:.1f}/200")
print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")

# Plot learning curves
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Returns
ax1.plot(metrics["iteration"], metrics["mean_return"], color="blue", label="Return")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax1.set_title("Learning Curve")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1.1, 1.1)

# 2. Loss (log scale)
ax2.plot(metrics["iteration"], np.abs(metrics["loss"]))
ax2.set_xlabel("Iteration")
ax2.set_ylabel("|Loss|")
ax2.set_title("REINFORCE Loss (log scale)")
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)

# 3. Actions
mean_act = metrics["mean_action"]
std_act = metrics["std_action"]
ax3.plot(metrics["iteration"], mean_act, label="Mean", color="green")
ax3.fill_between(metrics["iteration"],
                 [m-s for m,s in zip(mean_act, std_act)],
                 [m+s for m,s in zip(mean_act, std_act)],
                 alpha=0.3, color="green")
ax3.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label="Bounds")
ax3.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Action")
ax3.set_title("Action Distribution")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Sample episode trajectory
key, ep_key = jax.random.split(key)
episode = collect_episode(policy, step, reset_env, ep_key)
angles = episode.states[:, 0]
heights = jax.numpy.cos(angles)
ax4.plot(heights, color="purple")
ax4.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label="Success threshold")
ax4.set_xlabel("Step")
ax4.set_ylabel("cos(θ)")
ax4.set_title("Sample Episode (1.0 = top, -1.0 = bottom)")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-1.1, 1.1)

plt.suptitle(f"Pendulum Training Results - Success Rate: {successes/50:.0%}")
plt.tight_layout()
plt.savefig("tests/outputs/pendulum_success_analysis.png", dpi=150)
print("\nSaved analysis to tests/outputs/pendulum_success_analysis.png")

# Save best episode animation
print("\n=== Creating Animation of Best Episode ===")
# Find best episode
best_reward = -float('inf')
best_episode = None
for i in range(10):
    key, ep_key = jax.random.split(key)
    episode = collect_episode(policy, step, reset_env, ep_key)
    if episode.total_reward > best_reward:
        best_reward = episode.total_reward
        best_episode = episode

print(f"Best episode reward: {best_reward:.3f}")

try:
    from src.viz.pendulum import PendulumVisualizer
    viz = PendulumVisualizer()
    animation = viz.create_animation(
        [s for s in best_episode.states],
        [a for a in best_episode.actions], 
        [r for r in best_episode.rewards],
    )
    animation.save("tests/outputs/pendulum_success.mp4", writer="ffmpeg", fps=20)
    print("Saved animation to tests/outputs/pendulum_success.mp4")
except Exception as e:
    print(f"Animation failed: {e}")