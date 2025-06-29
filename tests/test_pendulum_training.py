# Test pendulum training with REINFORCE using JIT/vmap optimization

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
import pytest
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env
from src.train import collect_episode, collect_episodes, train_step
from src.baseline import BaselineState, update_baseline, compute_advantages
from src.viz.pendulum import PendulumVisualizer
import matplotlib.pyplot as plt


def train_pendulum(
    n_iterations: int = 100,
    episodes_per_iter: int = 32,
    learning_rate: float = 3e-3,
    use_baseline: bool = True,
    use_parallel: bool = True,
    seed: int = 42,
    verbose: bool = True,
):
    # Initialize
    key = jax.random.PRNGKey(seed)
    policy = GaussianPolicy(obs_dim=2, action_dim=1, hidden_dim=64)
    optimizer = nnx.Optimizer(policy, optax.adam(learning_rate))
    baseline_state = BaselineState(mean=jnp.array(0.0), n_samples=0)
    
    # Tracking
    metrics = {
        "iteration": [],
        "mean_return": [],
        "std_return": [],
        "loss": [],
        "baseline_value": [],
        "time_per_iter": [],
    }
    
    if verbose:
        print("Training pendulum swing-up with REINFORCE")
        print(f"Episodes per iteration: {episodes_per_iter}")
        print(f"Using {'parallel' if use_parallel else 'sequential'} episode collection")
        print(f"Using baseline: {use_baseline}\n")
    
    start_time = time.time()
    
    for i in range(n_iterations):
        iter_start = time.time()
        
        # Collect episodes
        key, subkey = jax.random.split(key)
        
        if use_parallel:
            # Collect all episodes in parallel with vmap
            episode_data = collect_episodes(
                policy, step, reset_env, subkey, episodes_per_iter, max_steps=200
            )
            all_states = episode_data.states
            all_actions = episode_data.actions
            all_returns = episode_data.returns
            episode_returns = episode_data.total_reward  # This is mean across episodes
        else:
            # Sequential collection (for comparison)
            episode_results = []
            for j in range(episodes_per_iter):
                key, episode_key = jax.random.split(key)
                episode = collect_episode(policy, step, reset_env, episode_key, max_steps=200)
                episode_results.append(episode)
            
            # Aggregate data
            all_states = jnp.concatenate([ep.states for ep in episode_results])
            all_actions = jnp.concatenate([ep.actions for ep in episode_results])
            all_returns = jnp.concatenate([ep.returns for ep in episode_results])
            episode_returns = jnp.mean(jnp.array([ep.total_reward for ep in episode_results]))
        
        # Compute advantages
        if use_baseline:
            advantages = compute_advantages(all_returns, baseline_state.mean)
            baseline_state = update_baseline(baseline_state, all_returns)
        else:
            advantages = all_returns
        
        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # Update policy
        loss = train_step(policy, optimizer, all_states, all_actions, advantages)
        
        iter_time = time.time() - iter_start
        
        # Track metrics
        metrics["iteration"].append(i)
        metrics["mean_return"].append(float(episode_returns))
        metrics["std_return"].append(0.0)  # Would need to track individual episodes
        metrics["loss"].append(float(loss))
        metrics["baseline_value"].append(float(baseline_state.mean))
        metrics["time_per_iter"].append(iter_time)
        
        if verbose and (i % 10 == 0 or i == n_iterations - 1):
            print(
                f"Iter {i:3d} | "
                f"Return: {episode_returns:7.2f} | "
                f"Loss: {loss:7.4f} | "
                f"Baseline: {baseline_state.mean:7.2f} | "
                f"Time: {iter_time:5.3f}s"
            )
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Average time per iteration: {total_time/n_iterations:.3f}s")
        print(f"Final return: {metrics['mean_return'][-1]:.2f}")
    
    return {
        "metrics": metrics,
        "policy": policy,
        "total_time": total_time,
    }


def evaluate_policy(policy, n_episodes=5, render=True):
    key = jax.random.PRNGKey(123)
    
    if render:
        viz = PendulumVisualizer()
    
    total_rewards = []
    
    for i in range(n_episodes):
        key, episode_key = jax.random.split(key)
        episode = collect_episode(policy, step, reset_env, episode_key, max_steps=200)
        total_rewards.append(float(episode.total_reward))
        
        if render and i == 0:  # Render first episode
            print(f"\nRendering episode with reward: {episode.total_reward:.2f}")
            animation = viz.create_animation(
                [s for s in episode.states],
                [a for a in episode.actions],
                [r for r in episode.rewards],
            )
            animation.save("tests/outputs/pendulum_trained.mp4", writer="ffmpeg", fps=20)
            print("Saved animation to tests/outputs/pendulum_trained.mp4")
    
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"Mean reward: {jnp.mean(jnp.array(total_rewards)):.2f}")
    print(f"Std reward: {jnp.std(jnp.array(total_rewards)):.2f}")


def plot_training_curves(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Returns
    ax1.plot(metrics["iteration"], metrics["mean_return"], label="Mean Return")
    ax1.plot(metrics["iteration"], metrics["baseline_value"], label="Baseline", alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Return")
    ax1.set_title("Training Returns")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(metrics["iteration"], metrics["loss"])
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("REINFORCE Loss")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("tests/outputs/pendulum_training_curves.png", dpi=150)
    print("\nSaved training curves to tests/outputs/pendulum_training_curves.png")


def compare_performance():
    print("=== Performance Comparison ===\n")
    
    # Train with sequential collection
    print("1. Sequential episode collection:")
    result_seq = train_pendulum(
        n_iterations=20,
        episodes_per_iter=16,
        use_parallel=False,
        verbose=False
    )
    seq_time = result_seq["total_time"]
    seq_return = result_seq["metrics"]["mean_return"][-1]
    print(f"   Time: {seq_time:.2f}s")
    print(f"   Final return: {seq_return:.2f}")
    
    # Train with parallel collection
    print("\n2. Parallel episode collection (vmap):")
    result_par = train_pendulum(
        n_iterations=20,
        episodes_per_iter=16,
        use_parallel=True,
        verbose=False
    )
    par_time = result_par["total_time"]
    par_return = result_par["metrics"]["mean_return"][-1]
    print(f"   Time: {par_time:.2f}s")
    print(f"   Final return: {par_return:.2f}")
    
    print(f"\n3. Speedup: {seq_time/par_time:.1f}x")


@pytest.mark.slow
def test_pendulum_training_performance():
    # Train with sequential collection
    result_seq = train_pendulum(
        n_iterations=10,
        episodes_per_iter=8,
        use_parallel=False,
        verbose=False
    )
    seq_time = result_seq["total_time"]
    
    # Train with parallel collection
    result_par = train_pendulum(
        n_iterations=10,
        episodes_per_iter=8,
        use_parallel=True,
        verbose=False
    )
    par_time = result_par["total_time"]
    
    # Parallel should be faster
    assert par_time < seq_time, f"Parallel ({par_time:.2f}s) not faster than sequential ({seq_time:.2f}s)"
    assert seq_time / par_time > 1.2, f"Speedup only {seq_time/par_time:.1f}x, expected >1.2x"


@pytest.mark.slow
def test_pendulum_training_convergence():
    result = train_pendulum(
        n_iterations=50,
        episodes_per_iter=16,
        use_parallel=True,
        use_baseline=True,
        verbose=False,
    )
    
    metrics = result["metrics"]
    initial_return = metrics["mean_return"][0]
    final_return = metrics["mean_return"][-1]
    
    # Should show some improvement (even if not solving the task)
    assert final_return > initial_return + 10, (
        f"No significant improvement: {initial_return:.1f} -> {final_return:.1f}"
    )
    
    # Save training curves for inspection
    plot_training_curves(metrics)


@pytest.mark.slow 
def test_pendulum_visualization():
    # Quick training run
    result = train_pendulum(
        n_iterations=20,
        episodes_per_iter=16,
        use_parallel=True,
        verbose=False,
    )
    
    # Evaluate and render
    evaluate_policy(result["policy"], n_episodes=1, render=True)