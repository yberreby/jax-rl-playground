#!/usr/bin/env python3
"""Optimized pendulum training for 100% success rate."""

import time
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import numpy as np

from src.policy import GaussianPolicy
from src.pendulum import step, reset_env, MAX_EPISODE_STEPS
from src.train import collect_episodes
from src.advantage_normalizer import (
    init_running_stats,
    update_running_stats,
    normalize_advantages,
)
from src.critic import ValueFunction, update_critic, compute_critic_advantages
from src.viz.pendulum import PendulumVisualizer


def train_pendulum_optimized(
    learning_rate=3e-4,  # Lower base LR
    batch_size=2048,  # Larger batch for stability
    hidden_dim=256,  # Larger network
    n_iterations=500,  # More iterations
    use_critic=True,
    entropy_weight=0.01,  # Small entropy for exploration
    visualize_every=100,
    output_dir="tests/outputs/pendulum_optimized",
    seed=42,
    verbose=True,
):
    """Optimized training configuration for pendulum swing-up."""

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=== Optimized Pendulum Training ===")
        print(f"Config: LR={learning_rate}, Batch={batch_size}, Hidden={hidden_dim}")
        print(f"Iterations: {n_iterations}, Entropy: {entropy_weight}")

    # Initialize policy with larger network
    policy = GaussianPolicy(2, 1, hidden_dim=hidden_dim)

    # More conservative learning rate schedule
    warmup_steps = int(0.3 * n_iterations)  # 30% warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=n_iterations,
        end_value=learning_rate * 0.01,  # Decay to 1% of initial
    )

    # Optimizer with more conservative gradient clipping
    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(5.0),  # Lower clip threshold
            optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-7),
        ),
    )

    # Initialize critic
    critic = ValueFunction(obs_dim=2, hidden_dim=hidden_dim)
    critic_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate * 2.0,  # 2x policy LR
        warmup_steps=warmup_steps,
        decay_steps=n_iterations,
        end_value=learning_rate * 0.02,
    )
    critic_optimizer = nnx.Optimizer(
        critic, optax.chain(optax.clip_by_global_norm(5.0), optax.adam(critic_schedule))
    )

    # Training state
    key = jax.random.PRNGKey(seed)
    advantage_stats = init_running_stats()

    # Track metrics
    best_return = -float("inf")
    success_episodes = 0
    recent_returns = []

    start_time = time.time()

    for i in range(n_iterations):
        # Collect episodes
        collect_key, key = jax.random.split(key)
        episode_batch = collect_episodes(
            policy, step, reset_env, collect_key, batch_size
        )

        current_return = float(episode_batch.total_reward)
        recent_returns.append(current_return)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        # Count successful episodes (positive return = mostly upright)
        if current_return > 0.5:
            success_episodes += 1

        # Update best
        if current_return > best_return:
            best_return = current_return

        # Compute advantages with critic
        update_critic(
            critic, critic_optimizer, episode_batch.states, episode_batch.returns
        )
        advantages = compute_critic_advantages(
            critic, episode_batch.states, episode_batch.returns
        )

        # Normalize advantages
        advantage_stats = update_running_stats(advantage_stats, advantages)
        if i >= 10:  # Start normalizing after some data
            advantages = normalize_advantages(advantages, advantage_stats)

        # Train step with entropy regularization
        def loss_fn(policy):
            log_probs = policy.log_prob(episode_batch.states, episode_batch.actions)
            pg_loss = -jnp.mean(log_probs * advantages)

            # Entropy bonus for exploration
            std = jnp.exp(policy.log_std.value)
            entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * std**2)
            entropy_loss = -entropy_weight * jnp.mean(entropy)

            return pg_loss + entropy_loss

        loss, grads = nnx.value_and_grad(loss_fn)(policy)

        # Track gradient norm
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(g.value if hasattr(g, "value") else g) for g in grad_leaves
        ]
        total_grad_norm = jnp.sqrt(sum(n**2 for n in grad_norms))

        optimizer.update(grads)

        # Progress reporting
        if verbose and i % 20 == 0:
            recent_avg = np.mean(recent_returns)
            print(
                f"Iter {i:3d} | Return: {current_return:6.3f} | "
                f"Recent avg: {recent_avg:6.3f} | Best: {best_return:6.3f} | "
                f"GradNorm: {total_grad_norm:6.2f}"
            )

        # Generate visualization
        if (i + 1) % visualize_every == 0 or i == n_iterations - 1:
            # Deterministic evaluation
            saved_std = policy.log_std.value
            policy.log_std.value = jnp.full_like(saved_std, -10.0)

            viz_key = jax.random.PRNGKey(seed + i + 1000)
            viz_episode = collect_episodes(policy, step, reset_env, viz_key, 1)

            policy.log_std.value = saved_std

            # Create video
            viz = PendulumVisualizer(dark_mode=True)
            video_path = output_dir / f"episode_{i + 1:03d}.mp4"
            states_list = [viz_episode.states[j] for j in range(MAX_EPISODE_STEPS)]
            actions_list = [viz_episode.actions[j] for j in range(MAX_EPISODE_STEPS)]
            rewards_list = [viz_episode.rewards[j] for j in range(MAX_EPISODE_STEPS)]
            viz.create_animation(
                states=states_list,
                actions=actions_list,
                rewards=rewards_list,
                filename=str(video_path),
            )

            if verbose:
                print(f"  → Saved video for iteration {i + 1}")

        # Early stopping if we achieve consistent success
        if len(recent_returns) >= 50 and np.mean(recent_returns) > 0.8:
            if verbose:
                print(f"\n✓ Achieved consistent success at iteration {i}!")
            break

    training_time = time.time() - start_time

    # Final evaluation
    eval_returns = []
    eval_key = jax.random.PRNGKey(seed + 2000)

    for _ in range(100):
        ep_key, eval_key = jax.random.split(eval_key)
        episode_batch = collect_episodes(policy, step, reset_env, ep_key, 1)
        eval_returns.append(float(episode_batch.total_reward))

    final_performance = np.mean(eval_returns)
    success_rate = sum(1 for r in eval_returns if r > 0.5) / len(eval_returns)

    if verbose:
        print("\n=== Final Results ===")
        print(f"Training time: {training_time:.1f}s")
        print(f"Final performance: {final_performance:.3f}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Best return: {best_return:.3f}")
        print(f"Final policy std: {float(jnp.exp(policy.log_std.value[0])):.3f}")

    return {
        "final_performance": final_performance,
        "success_rate": success_rate,
        "best_return": best_return,
        "training_time": training_time,
        "output_dir": str(output_dir),
    }


def test_optimized_training():
    """Test optimized training configuration."""
    print("Running optimized pendulum training...")

    results = train_pendulum_optimized(
        learning_rate=3e-4,
        batch_size=2048,
        hidden_dim=256,
        n_iterations=500,
        entropy_weight=0.01,
        visualize_every=100,
        seed=42,
    )

    # We want at least 80% success rate
    assert results["success_rate"] > 0.8, (
        f"Success rate too low: {results['success_rate']}"
    )
    assert results["final_performance"] > 0.5, (
        f"Performance too low: {results['final_performance']}"
    )

    print("\n✓ Optimized training test completed!")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Check outputs in: {results['output_dir']}")

    return results


if __name__ == "__main__":
    test_optimized_training()
