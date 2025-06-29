#!/usr/bin/env python3
"""Debug why pendulum isn't learning to swing up properly."""

import jax
import jax.numpy as jnp
from src.policy import GaussianPolicy
from src.pendulum import step, reset_env
from src.train import train, collect_episode

# Train with different configs
configs = [
    {"lr": 3e-3, "episodes": 32, "iters": 300},
    {"lr": 1e-2, "episodes": 64, "iters": 200},
    {"lr": 3e-2, "episodes": 32, "iters": 200},
]

print("=== Testing Different Configurations ===")
for i, config in enumerate(configs):
    print(f"\nConfig {i+1}: lr={config['lr']}, episodes={config['episodes']}")
    
    policy = GaussianPolicy(2, 1, hidden_dim=128)
    
    # Initial performance
    key = jax.random.PRNGKey(42)
    init_ep = collect_episode(policy, step, reset_env, key)
    print(f"  Initial: {init_ep.total_reward:.3f}")
    
    # Train
    metrics = train(
        policy, step, reset_env,
        n_iterations=config["iters"],
        episodes_per_iter=config["episodes"],
        learning_rate=config["lr"],
        use_baseline=True,
        verbose=False,
    )
    
    # Final performance - collect multiple episodes
    final_rewards = []
    top_counts = []
    for j in range(10):
        key, ep_key = jax.random.split(key)
        episode = collect_episode(policy, step, reset_env, ep_key)
        final_rewards.append(float(episode.total_reward))
        
        # Count time at top
        heights = jnp.cos(episode.states[:, 0])
        top_count = float(jnp.sum(heights > 0.9))
        top_counts.append(top_count)
    
    print(f"  Final avg: {jnp.mean(jnp.array(final_rewards)):.3f} ± {jnp.std(jnp.array(final_rewards)):.3f}")
    print(f"  Steps at top: {jnp.mean(jnp.array(top_counts)):.1f} ± {jnp.std(jnp.array(top_counts)):.1f}")
    print(f"  Best during training: {max(metrics['mean_return']):.3f}")
    
    # Check if it learned to balance or swing up
    key, ep_key = jax.random.split(key)
    episode = collect_episode(policy, step, reset_env, ep_key)
    
    # Analyze trajectory
    angles = episode.states[:, 0]
    velocities = episode.states[:, 1]
    actions = episode.actions[:, 0]
    
    # Does it swing up from bottom?
    bottom_start = jnp.abs(angles[0] - jnp.pi) < 0.5
    reaches_top = jnp.any(jnp.cos(angles) > 0.9)
    
    # Energy analysis
    heights = -jnp.cos(angles)  # 0 at top, 2 at bottom
    kinetic = 0.5 * velocities**2
    potential = 10.0 * heights  # g * h
    total_energy = kinetic + potential
    energy_gain = total_energy[-1] - total_energy[0]
    
    print(f"  Starts at bottom: {bottom_start}")
    print(f"  Reaches top: {reaches_top}")
    print(f"  Energy gain: {energy_gain:.1f}")
    print(f"  Action range: [{actions.min():.2f}, {actions.max():.2f}]")