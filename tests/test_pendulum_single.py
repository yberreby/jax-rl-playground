#!/usr/bin/env python3
"""Single-run pendulum training with comprehensive metrics and visualizations."""

import time
import csv
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
import numpy as np

from src.policy import GaussianPolicy
from src.pendulum import step, reset_env, MAX_EPISODE_STEPS
from src.train import collect_episodes
from src.baseline import BaselineState, update_baseline, compute_advantages
from src.advantage_normalizer import (
    init_running_stats,
    update_running_stats,
    normalize_advantages,
)
from src.critic import ValueFunction, update_critic, compute_critic_advantages
from src.viz.pendulum import PendulumVisualizer


def compute_episode_metrics(states, rewards, actions, angle_threshold=0.1):
    """Compute detailed metrics for a single episode."""
    angles = jnp.array([s[0] for s in states])
    velocities = jnp.array([s[1] for s in states])

    # Upright metrics (angle close to 0)
    upright_mask = jnp.abs(angles) < angle_threshold
    upright_fraction = jnp.mean(upright_mask)

    # Find first swing-up time (if any)
    swing_up_time = None
    if jnp.any(upright_mask):
        swing_up_time = int(jnp.argmax(upright_mask))

    # Stability: std of angle when upright
    angle_stability = (
        jnp.std(angles[upright_mask]) if jnp.any(upright_mask) else jnp.inf
    )

    # Energy metrics
    kinetic_energy = 0.5 * velocities**2
    potential_energy = 1.0 - jnp.cos(angles)
    total_energy = kinetic_energy + potential_energy

    # Action metrics
    action_magnitude = jnp.abs(actions).mean()
    action_changes = jnp.abs(jnp.diff(actions.squeeze())).mean()

    return {
        "upright_fraction": float(upright_fraction),
        "swing_up_time": swing_up_time,
        "angle_stability": float(angle_stability),
        "mean_energy": float(total_energy.mean()),
        "energy_std": float(total_energy.std()),
        "action_magnitude": float(action_magnitude),
        "action_smoothness": float(action_changes),
        "final_angle": float(angles[-1]),
        "final_velocity": float(velocities[-1]),
    }


def plot_training_curves(output_dir, metrics_history):
    """Create comprehensive training plots."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    iterations = range(len(metrics_history["returns"]))

    # 1. Episode returns
    ax = axes[0]
    ax.plot(iterations, metrics_history["returns"], "b-", alpha=0.6)
    # Only plot moving average if we have enough data
    if len(metrics_history["returns"]) >= 10:
        moving_avg = np.convolve(metrics_history["returns"], np.ones(10) / 10, "valid")
        ax.plot(
            range(9, len(metrics_history["returns"])), moving_avg, "b-", linewidth=2
        )
    ax.set_title("Episode Returns")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)

    # 2. Success rate (moving average)
    ax = axes[1]
    window = min(20, max(5, len(metrics_history["upright_fractions"]) // 4))
    success_rate = []
    if len(metrics_history["upright_fractions"]) >= window:
        for i in range(window, len(metrics_history["upright_fractions"])):
            rate = np.mean(
                [f > 0.8 for f in metrics_history["upright_fractions"][i - window : i]]
            )
            success_rate.append(rate)
        ax.plot(
            range(window, len(metrics_history["upright_fractions"])), success_rate, "g-"
        )
    ax.set_title(f"Success Rate ({window}-episode window)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # 3. Gradient norms
    ax = axes[2]
    ax.semilogy(iterations, metrics_history["grad_norms"], "r-", alpha=0.6)
    ax.set_title("Gradient Norms")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Norm (log scale)")
    ax.grid(True, alpha=0.3)

    # 4. Policy std
    ax = axes[3]
    ax.plot(iterations, metrics_history["policy_stds"], "m-")
    ax.set_title("Policy Std Dev")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Std")
    ax.grid(True, alpha=0.3)

    # 5. Upright fraction
    ax = axes[4]
    ax.plot(iterations, metrics_history["upright_fractions"], "c-", alpha=0.6)
    ax.set_title("Fraction of Time Upright")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Upright Fraction")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # 6. Swing-up times
    ax = axes[5]
    valid_swingups = [
        (i, t) for i, t in enumerate(metrics_history["swing_up_times"]) if t is not None
    ]
    if valid_swingups:
        indices, times = zip(*valid_swingups)
        ax.scatter(indices, times, alpha=0.5, s=20)
    ax.set_title("Swing-up Times")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Steps to Swing-up")
    ax.set_ylim(0, MAX_EPISODE_STEPS)
    ax.grid(True, alpha=0.3)

    # 7. Energy
    ax = axes[6]
    ax.plot(iterations, metrics_history["mean_energies"], "y-", label="Mean")
    ax.fill_between(
        iterations,
        np.array(metrics_history["mean_energies"])
        - np.array(metrics_history["energy_stds"]),
        np.array(metrics_history["mean_energies"])
        + np.array(metrics_history["energy_stds"]),
        alpha=0.3,
    )
    ax.set_title("Total Energy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Action metrics
    ax = axes[7]
    ax.plot(
        iterations,
        metrics_history["action_magnitudes"],
        "orange",
        label="Magnitude",
        alpha=0.7,
    )
    ax.plot(
        iterations,
        metrics_history["action_smoothness"],
        "brown",
        label="Smoothness",
        alpha=0.7,
    )
    ax.set_title("Action Metrics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Learning rate schedule (if applicable)
    ax = axes[8]
    ax.plot(iterations, metrics_history["learning_rates"], "purple")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def train_pendulum_with_metrics(
    learning_rate=5e-4,
    batch_size=8192,
    hidden_dim=64,
    n_iterations=100,
    use_critic=True,  # Enable critic by default
    entropy_weight=0.0,
    visualize_every=25,
    output_dir="tests/outputs/pendulum_single",
    seed=0,
    verbose=True,
):
    """Train pendulum with comprehensive metrics and visualizations."""

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=== Pendulum Training with Rich Metrics ===")
        print(f"Config: LR={learning_rate}, Batch={batch_size}, Hidden={hidden_dim}")
        print(f"Iterations: {n_iterations}, Critic: {use_critic}")
        print(f"Output directory: {output_dir}\n")

    # Initialize policy
    policy = GaussianPolicy(2, 1, hidden_dim=hidden_dim)

    # Initialize optimizer with learning rate schedule with warmup
    warmup_steps = int(0.2 * n_iterations)  # 20% warmup for stability
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=n_iterations,
        end_value=learning_rate * 0.1,  # Final LR will be 10% of initial
    )

    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-7),
        ),
    )

    # Initialize critic if requested
    if use_critic:
        critic = ValueFunction(
            obs_dim=2, hidden_dim=hidden_dim
        )  # Match policy hidden dim
        # Critic learns faster with higher LR and its own schedule
        critic_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate * 1.5,  # Slightly higher than policy
            warmup_steps=warmup_steps,
            decay_steps=n_iterations,
            end_value=learning_rate * 0.3,
        )
        critic_optimizer = nnx.Optimizer(critic, optax.adam(critic_schedule))

    # Training state
    key = jax.random.PRNGKey(seed)
    baseline_state = BaselineState(mean=jnp.array(0.0), n_samples=0)
    advantage_stats = init_running_stats()

    # Metrics tracking
    metrics_history = {
        "returns": [],
        "grad_norms": [],
        "policy_stds": [],
        "upright_fractions": [],
        "swing_up_times": [],
        "angle_stabilities": [],
        "mean_energies": [],
        "energy_stds": [],
        "action_magnitudes": [],
        "action_smoothness": [],
        "learning_rates": [],
    }

    # CSV logging
    csv_path = output_dir / "metrics.csv"
    csv_columns = [
        "iteration",
        "return",
        "grad_norm",
        "policy_std",
        "upright_fraction",
        "swing_up_time",
        "angle_stability",
        "mean_energy",
        "action_magnitude",
        "action_smoothness",
        "learning_rate",
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

    # Training loop
    start_time = time.time()
    best_return = -float("inf")
    best_episode_data = None

    for i in range(n_iterations):
        # Collect episodes
        collect_key, key = jax.random.split(key)
        episode_batch = collect_episodes(
            policy, step, reset_env, collect_key, batch_size
        )

        # Compute detailed metrics for the first episode
        episode_metrics = compute_episode_metrics(
            episode_batch.states[:MAX_EPISODE_STEPS],  # First episode
            episode_batch.rewards[:MAX_EPISODE_STEPS],
            episode_batch.actions[:MAX_EPISODE_STEPS],
        )

        current_return = float(episode_batch.total_reward)
        current_lr = float(schedule(i))

        # Track best episode for visualization
        if current_return > best_return:
            best_return = current_return
            best_episode_data = {
                "states": episode_batch.states[:MAX_EPISODE_STEPS],
                "actions": episode_batch.actions[:MAX_EPISODE_STEPS],
                "rewards": episode_batch.rewards[:MAX_EPISODE_STEPS],
            }

        # Update metrics history
        metrics_history["returns"].append(current_return)
        metrics_history["policy_stds"].append(float(jnp.exp(policy.log_std.value[0])))
        metrics_history["upright_fractions"].append(episode_metrics["upright_fraction"])
        metrics_history["swing_up_times"].append(episode_metrics["swing_up_time"])
        metrics_history["angle_stabilities"].append(episode_metrics["angle_stability"])
        metrics_history["mean_energies"].append(episode_metrics["mean_energy"])
        metrics_history["energy_stds"].append(episode_metrics["energy_std"])
        metrics_history["action_magnitudes"].append(episode_metrics["action_magnitude"])
        metrics_history["action_smoothness"].append(
            episode_metrics["action_smoothness"]
        )
        metrics_history["learning_rates"].append(current_lr)

        # Compute advantages
        if use_critic:
            update_critic(
                critic, critic_optimizer, episode_batch.states, episode_batch.returns
            )
            advantages = compute_critic_advantages(
                critic, episode_batch.states, episode_batch.returns
            )
        else:
            advantages = compute_advantages(episode_batch.returns, baseline_state.mean)
            baseline_state = update_baseline(baseline_state, episode_batch.returns)

        # Normalize advantages
        advantage_stats = update_running_stats(advantage_stats, advantages)
        if i >= 5:
            advantages = normalize_advantages(advantages, advantage_stats)

        # Train step
        def loss_fn(policy):
            log_probs = policy.log_prob(episode_batch.states, episode_batch.actions)
            pg_loss = -jnp.mean(log_probs * advantages)

            if entropy_weight > 0:
                std = jnp.exp(policy.log_std.value)
                entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * std**2)
                entropy_loss = -entropy_weight * jnp.mean(entropy)
                return pg_loss + entropy_loss
            return pg_loss

        loss, grads = nnx.value_and_grad(loss_fn)(policy)

        # Track gradient norm
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms_batch = [
            jnp.linalg.norm(g.value if hasattr(g, "value") else g) for g in grad_leaves
        ]
        total_grad_norm = jnp.sqrt(sum(n**2 for n in grad_norms_batch))
        metrics_history["grad_norms"].append(float(total_grad_norm))

        optimizer.update(grads)

        # Log to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(
                {
                    "iteration": i,
                    "return": current_return,
                    "grad_norm": float(total_grad_norm),
                    "policy_std": metrics_history["policy_stds"][-1],
                    "upright_fraction": episode_metrics["upright_fraction"],
                    "swing_up_time": episode_metrics["swing_up_time"] or -1,
                    "angle_stability": episode_metrics["angle_stability"],
                    "mean_energy": episode_metrics["mean_energy"],
                    "action_magnitude": episode_metrics["action_magnitude"],
                    "action_smoothness": episode_metrics["action_smoothness"],
                    "learning_rate": current_lr,
                }
            )

        # Progress reporting
        if verbose and i % 10 == 0:
            print(
                f"Iter {i:3d} | Return: {current_return:6.3f} | "
                f"Upright: {episode_metrics['upright_fraction']:.2%} | "
                f"GradNorm: {total_grad_norm:6.2f} | "
                f"LR: {current_lr:.2e}"
            )

        # Generate visualizations periodically
        if (i + 1) % visualize_every == 0 or i == n_iterations - 1:
            # Collect a DETERMINISTIC test episode for visualization
            # Save current policy std and set to zero for deterministic eval
            saved_std = policy.log_std.value
            policy.log_std.value = jnp.full_like(
                saved_std, -10.0
            )  # Very small std ≈ deterministic

            viz_key = jax.random.PRNGKey(seed + i + 1000)
            viz_episode = collect_episodes(policy, step, reset_env, viz_key, 1)

            # Restore original std
            policy.log_std.value = saved_std

            # Create video
            viz = PendulumVisualizer(dark_mode=True)
            video_path = output_dir / f"episode_{i + 1:03d}.mp4"
            # Convert arrays to lists for visualization
            states_list = [
                viz_episode.states[j]
                for j in range(min(MAX_EPISODE_STEPS, len(viz_episode.states)))
            ]
            actions_list = [
                viz_episode.actions[j]
                for j in range(min(MAX_EPISODE_STEPS, len(viz_episode.actions)))
            ]
            rewards_list = [
                viz_episode.rewards[j]
                for j in range(min(MAX_EPISODE_STEPS, len(viz_episode.rewards)))
            ]
            viz.create_animation(
                states=states_list,
                actions=actions_list,
                rewards=rewards_list,
                filename=str(video_path),
            )

            # Create phase portrait
            phase_path = output_dir / f"phase_{i + 1:03d}.png"
            viz.create_phase_portrait(states=states_list, filename=str(phase_path))

            if verbose:
                print(f"  → Saved visualizations for iteration {i + 1}")

    training_time = time.time() - start_time

    # Final evaluation
    eval_returns = []
    eval_metrics = []
    eval_key = jax.random.PRNGKey(seed + 1000)

    for _ in range(50):
        ep_key, eval_key = jax.random.split(eval_key)
        episode_batch = collect_episodes(policy, step, reset_env, ep_key, 1)
        eval_returns.append(float(episode_batch.total_reward))

        ep_metrics = compute_episode_metrics(
            episode_batch.states[:200],
            episode_batch.rewards[:200],
            episode_batch.actions[:200],
        )
        eval_metrics.append(ep_metrics)

    final_performance = jnp.mean(jnp.array(eval_returns))
    performance_std = jnp.std(jnp.array(eval_returns))

    # Aggregate evaluation metrics
    success_rate = sum(1 for m in eval_metrics if m["upright_fraction"] > 0.8) / len(
        eval_metrics
    )
    mean_upright = np.mean([m["upright_fraction"] for m in eval_metrics])
    mean_stability = np.mean(
        [m["angle_stability"] for m in eval_metrics if m["angle_stability"] < np.inf]
    )

    # Generate final plots
    plot_training_curves(output_dir, metrics_history)

    # Create best episode video
    if best_episode_data:
        viz = PendulumVisualizer(dark_mode=True)
        # Convert to lists
        states_list = [
            best_episode_data["states"][j]
            for j in range(len(best_episode_data["states"]))
        ]
        actions_list = [
            best_episode_data["actions"][j]
            for j in range(len(best_episode_data["actions"]))
        ]
        rewards_list = [
            best_episode_data["rewards"][j]
            for j in range(len(best_episode_data["rewards"]))
        ]
        viz.create_animation(
            states=states_list,
            actions=actions_list,
            rewards=rewards_list,
            filename=str(output_dir / "best_episode.mp4"),
        )

    # Final report
    if verbose:
        print("\n=== Final Results ===")
        print(f"Training time: {training_time:.1f}s")
        print(f"Final performance: {final_performance:.3f} ± {performance_std:.3f}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Mean upright fraction: {mean_upright:.1%}")
        print(f"Mean stability (when upright): {mean_stability:.3f}")
        print(f"Best training return: {best_return:.3f}")
        print(f"Final policy std: {float(jnp.exp(policy.log_std.value[0])):.3f}")
        print(f"\nOutputs saved to: {output_dir}")

    return {
        "final_performance": float(final_performance),
        "performance_std": float(performance_std),
        "success_rate": success_rate,
        "mean_upright": mean_upright,
        "mean_stability": mean_stability,
        "best_return": best_return,
        "training_time": training_time,
        "metrics_history": metrics_history,
        "output_dir": str(output_dir),
    }


def test_single_run():
    """Test single run with best known hyperparameters."""
    print("Running single pendulum training with rich metrics...")

    results = train_pendulum_with_metrics(
        learning_rate=5e-4,  # Conservative LR
        batch_size=1024,  # Larger batch
        hidden_dim=128,  # Larger network
        n_iterations=200,  # More iterations
        use_critic=True,  # Use critic for lower variance
        entropy_weight=0.0,
        visualize_every=50,
        seed=42,
    )

    # Basic assertions (positive returns now mean upright)
    assert results["final_performance"] > -0.5, (
        f"Performance too low: {results['final_performance']}"
    )
    assert results["success_rate"] > 0.05, (
        f"Success rate too low: {results['success_rate']}"
    )

    print("\n✓ Single run test completed successfully!")
    print(f"  Check outputs in: {results['output_dir']}")

    return results


if __name__ == "__main__":
    test_single_run()
