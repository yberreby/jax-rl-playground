#!/usr/bin/env python3
"""Debug pendulum training with proper tracking."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.policy import GaussianPolicy, sample_actions
from src.pendulum import step, reset_env
from src.train import train, collect_episode

# Test if log probs are correct
print("=== Testing Log Prob Correction ===")
policy = GaussianPolicy(2, 1, hidden_dim=64)
obs = jnp.array([[0.0, 0.0], [jnp.pi, 0.0]])
key = jax.random.PRNGKey(42)

# Sample multiple times
all_actions = []
all_log_probs = []
for i in range(100):
    key, subkey = jax.random.split(key)
    actions, log_probs = sample_actions(policy, obs, subkey)
    all_actions.append(actions)
    all_log_probs.append(log_probs)

all_actions = jnp.stack(all_actions)
all_log_probs = jnp.stack(all_log_probs)

print(f"Action range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
print(f"Mean log prob: {all_log_probs.mean():.3f}")
print(f"Actions properly bounded: {(all_actions.min() >= -2.0) and (all_actions.max() <= 2.0)}")

# Train with detailed tracking
print("\n=== Training with Detailed Tracking ===")
policy = GaussianPolicy(2, 1, hidden_dim=128)

metrics = train(
    policy, step, reset_env,
    n_iterations=100,
    episodes_per_iter=32,
    learning_rate=1e-2,
    use_baseline=True,
    verbose=True,
)

# Plot comprehensive metrics
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# 1. Returns and baseline
ax = axes[0]
ax.plot(metrics["iteration"], metrics["mean_return"], label="Return", color="blue")
ax.plot(metrics["iteration"], metrics["baseline_value"], label="Baseline", color="red", alpha=0.7)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Returns vs Baseline")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Loss
ax = axes[1]
ax.plot(metrics["iteration"], metrics["loss"])
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("REINFORCE Loss")
ax.set_yscale("symlog")
ax.grid(True, alpha=0.3)

# 3. Advantages
ax = axes[2]
mean_adv = metrics["mean_advantage"]
std_adv = metrics["std_advantage"]
ax.plot(metrics["iteration"], mean_adv, label="Mean")
ax.fill_between(metrics["iteration"], 
                 [m-s for m,s in zip(mean_adv, std_adv)],
                 [m+s for m,s in zip(mean_adv, std_adv)], 
                 alpha=0.3)
ax.set_xlabel("Iteration")
ax.set_ylabel("Advantage")
ax.set_title("Advantages")
ax.grid(True, alpha=0.3)

# 4. Log probs
ax = axes[3]
ax.plot(metrics["iteration"], metrics["mean_log_prob"])
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean Log Prob")
ax.set_title("Log Probabilities")
ax.grid(True, alpha=0.3)

# 5. Actions
ax = axes[4]
mean_act = metrics["mean_action"]
std_act = metrics["std_action"]
ax.plot(metrics["iteration"], mean_act, label="Mean")
ax.fill_between(metrics["iteration"],
                 [m-s for m,s in zip(mean_act, std_act)],
                 [m+s for m,s in zip(mean_act, std_act)],
                 alpha=0.3)
ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
ax.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Action")
ax.set_title("Action Distribution")
ax.grid(True, alpha=0.3)

# 6. Return improvement
ax = axes[5]
returns = metrics["mean_return"]
improvement = [r - returns[0] for r in returns]
ax.plot(metrics["iteration"], improvement)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel("Iteration")
ax.set_ylabel("Improvement")
ax.set_title("Return Improvement from Initial")
ax.grid(True, alpha=0.3)

# 7. Baseline tracking error
ax = axes[6]
baseline_errors = [abs(r - b) for r, b in zip(metrics["mean_return"], metrics["baseline_value"])]
ax.plot(metrics["iteration"], baseline_errors)
ax.set_xlabel("Iteration")
ax.set_ylabel("|Return - Baseline|")
ax.set_title("Baseline Tracking Error")
ax.grid(True, alpha=0.3)

# 8. Episode analysis
ax = axes[7]
# Collect one episode for analysis
key = jax.random.PRNGKey(123)
episode = collect_episode(policy, step, reset_env, key)
angles = episode.states[:, 0]
heights = jnp.cos(angles)
ax.plot(heights, label="Height (cos(θ))")
ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label="Near top")
ax.set_xlabel("Step")
ax.set_ylabel("Height")
ax.set_title("Final Episode Trajectory")
ax.legend()
ax.grid(True, alpha=0.3)

# 9. Policy parameters
ax = axes[8]
ax.text(0.1, 0.9, "Final Statistics:", transform=ax.transAxes, fontsize=12, weight='bold')
ax.text(0.1, 0.7, f"Best return: {max(returns):.3f}", transform=ax.transAxes)
ax.text(0.1, 0.6, f"Final return: {returns[-1]:.3f}", transform=ax.transAxes)
ax.text(0.1, 0.5, f"Final baseline: {metrics['baseline_value'][-1]:.3f}", transform=ax.transAxes)
ax.text(0.1, 0.4, f"Final loss: {metrics['loss'][-1]:.2e}", transform=ax.transAxes)
ax.text(0.1, 0.3, f"Log std: {float(policy.log_std.value[0]):.3f}", transform=ax.transAxes)
ax.text(0.1, 0.2, f"Std: {float(jnp.exp(policy.log_std.value[0])):.3f}", transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig("tests/outputs/pendulum_debug_tracking.png", dpi=150)
print("\nSaved comprehensive tracking plot to tests/outputs/pendulum_debug_tracking.png")

# Final evaluation
print("\n=== Final Evaluation ===")
successes = 0
total_steps_at_top = 0
for i in range(20):
    key, ep_key = jax.random.split(key)
    episode = collect_episode(policy, step, reset_env, ep_key)
    heights = jnp.cos(episode.states[:, 0])
    steps_at_top = float(jnp.sum(heights > 0.9))
    total_steps_at_top += steps_at_top
    if steps_at_top > 100:  # More than half the episode
        successes += 1

print(f"Success rate (>100 steps at top): {successes}/20 = {successes/20:.0%}")
print(f"Average steps at top: {total_steps_at_top/20:.1f}/200")