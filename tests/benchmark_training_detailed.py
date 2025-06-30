#!/usr/bin/env python3
"""Detailed benchmark of training loop to find bottlenecks."""

import time
import jax
import jax.numpy as jnp
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env
from src.train import collect_episodes, train_step
from src.critic import ValueFunction, update_critic, compute_critic_advantages
from src.advantage_normalizer import normalize_advantages
from flax import nnx
import optax


def benchmark_training_components():
    """Benchmark each component of the training loop."""
    print("=== Detailed Training Benchmark ===")
    
    # Initialize everything
    policy = GaussianPolicy(obs_dim=2, action_dim=1, hidden_dim=128, n_hidden_layers=2)
    critic = ValueFunction(obs_dim=2, hidden_dim=64)
    
    optimizer = nnx.Optimizer(policy, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    ))
    critic_optimizer = nnx.Optimizer(critic, optax.adam(2e-4))
    
    key = jax.random.PRNGKey(42)
    
    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    warmup_start = time.time()
    
    key, subkey = jax.random.split(key)
    episode_batch = collect_episodes(policy, step, reset_env, subkey, n_episodes=32)
    _ = update_critic(critic, critic_optimizer, episode_batch.states, episode_batch.returns)
    _ = train_step(policy, optimizer, episode_batch.states, episode_batch.actions, episode_batch.returns)
    
    # Force completion
    jax.block_until_ready(episode_batch.states)
    warmup_time = time.time() - warmup_start
    print(f"Warmup time: {warmup_time:.2f}s")
    
    # Benchmark with different batch sizes
    for n_episodes in [32, 128, 512, 1024]:
        print(f"\n=== Batch size: {n_episodes} episodes ===")
        times = []
        
        for iteration in range(5):  # 5 iterations to average
            iter_times = {}
            total_start = time.time()
            
            # 1. Episode collection
            start = time.time()
            key, subkey = jax.random.split(key)
            episode_batch = collect_episodes(policy, step, reset_env, subkey, n_episodes=n_episodes)
            jax.block_until_ready(episode_batch.states)  # Force completion
            iter_times['collect'] = time.time() - start
            
            # Extract data
            all_states = episode_batch.states
            all_actions = episode_batch.actions
            all_returns = episode_batch.returns
            
            # 2. Critic update
            start = time.time()
            critic_loss = update_critic(critic, critic_optimizer, all_states, all_returns)
            jax.block_until_ready(critic_loss)
            iter_times['critic_update'] = time.time() - start
            
            # 3. Compute advantages
            start = time.time()
            advantages = compute_critic_advantages(critic, all_states, all_returns)
            jax.block_until_ready(advantages)
            iter_times['compute_advantages'] = time.time() - start
            
            # 4. Normalize advantages
            start = time.time()
            normalized_advantages = normalize_advantages(advantages)
            jax.block_until_ready(normalized_advantages)
            iter_times['normalize'] = time.time() - start
            
            # 5. Policy update
            start = time.time()
            loss, grad_norm, grad_var = train_step(
                policy, optimizer, all_states, all_actions, normalized_advantages
            )
            jax.block_until_ready(loss)
            iter_times['policy_update'] = time.time() - start
            
            # 6. Metrics computation (the suspected bottleneck!)
            start = time.time()
            # Computing metrics one by one with device-to-host transfers
            _ = float(episode_batch.total_reward)
            _ = float(loss)
            _ = float(jnp.mean(advantages))
            _ = float(jnp.std(advantages))
            _ = float(jnp.mean(episode_batch.log_probs))
            _ = float(jnp.mean(episode_batch.actions))
            _ = float(jnp.std(episode_batch.actions))
            _ = float(grad_norm)
            _ = float(grad_var)
            _ = float(jnp.max(episode_batch.actions))
            _ = float(jnp.min(episode_batch.actions))
            iter_times['metrics'] = time.time() - start
            
            # 7. Test single vs batched metric transfer
            start = time.time()
            # Batched transfer - create array then convert once
            metric_array = jnp.array([
                episode_batch.total_reward,
                loss,
                jnp.mean(advantages),
                jnp.std(advantages),
                jnp.mean(episode_batch.log_probs),
                jnp.mean(episode_batch.actions),
                jnp.std(episode_batch.actions),
                grad_norm,
                grad_var,
                jnp.max(episode_batch.actions),
                jnp.min(episode_batch.actions),
            ])
            _ = list(float(x) for x in metric_array)
            iter_times['metrics_batched'] = time.time() - start
            
            iter_times['total'] = time.time() - total_start
            times.append(iter_times)
        
        # Compute averages
        avg_times = {}
        for key in times[0].keys():
            avg_times[key] = sum(t[key] for t in times) / len(times)
        
        # Print results
        print(f"Total time per iteration: {avg_times['total']:.3f}s")
        print("Breakdown:")
        for component, t in avg_times.items():
            if component != 'total':
                pct = t / avg_times['total'] * 100
                print(f"  {component:20s}: {t:.3f}s ({pct:4.1f}%)")
        
        # Compute throughput
        total_steps = n_episodes * 400  # 400 steps per episode
        steps_per_sec = total_steps / avg_times['total']
        print(f"Throughput: {steps_per_sec:,.0f} steps/sec")
        
        # Compare metrics approaches
        print("\nMetrics comparison:")
        print(f"  Individual transfers: {avg_times['metrics']:.3f}s")
        print(f"  Batched transfer:     {avg_times['metrics_batched']:.3f}s")
        speedup = avg_times['metrics'] / avg_times['metrics_batched']
        print(f"  Speedup: {speedup:.1f}x")


def benchmark_minimal_loop():
    """Benchmark minimal training loop without metrics."""
    print("\n\n=== Minimal Training Loop (no metrics) ===")
    
    policy = GaussianPolicy(obs_dim=2, action_dim=1, hidden_dim=128, n_hidden_layers=2)
    critic = ValueFunction(obs_dim=2, hidden_dim=64)
    
    optimizer = nnx.Optimizer(policy, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    ))
    critic_optimizer = nnx.Optimizer(critic, optax.adam(2e-4))
    
    key = jax.random.PRNGKey(42)
    
    # Warmup
    key, subkey = jax.random.split(key)
    episode_batch = collect_episodes(policy, step, reset_env, subkey, n_episodes=512)
    jax.block_until_ready(episode_batch.states)
    
    # Time 10 iterations
    start_time = time.time()
    for i in range(10):
        # Collect
        key, subkey = jax.random.split(key)
        episode_batch = collect_episodes(policy, step, reset_env, subkey, n_episodes=512)
        
        # Update critic and compute advantages
        update_critic(critic, critic_optimizer, episode_batch.states, episode_batch.returns)
        advantages = compute_critic_advantages(critic, episode_batch.states, episode_batch.returns)
        
        # Normalize and update policy
        normalized_advantages = normalize_advantages(advantages)
        loss, _, _ = train_step(policy, optimizer, episode_batch.states, episode_batch.actions, normalized_advantages)
        
        # Force completion
        jax.block_until_ready(loss)
        
        if i == 0:
            first_iter_time = time.time() - start_time
            print(f"First iteration (with JIT): {first_iter_time:.3f}s")
    
    total_time = time.time() - start_time
    per_iter = (total_time - first_iter_time) / 9  # Exclude first iteration
    
    print(f"Average per iteration (after JIT): {per_iter:.3f}s")
    print(f"Throughput: {512 * 400 / per_iter:,.0f} steps/sec")


if __name__ == "__main__":
    benchmark_training_components()
    benchmark_minimal_loop()