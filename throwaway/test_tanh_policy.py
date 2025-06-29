#!/usr/bin/env python3
"""Test proper tanh squashing for bounded actions."""

import jax
import jax.numpy as jnp
from flax import nnx
from src.distributions import gaussian_log_prob
from src.policy import GaussianPolicy
from src.pendulum import MAX_TORQUE

# Proper tanh-squashed Gaussian policy
@nnx.jit  
def sample_actions_with_tanh_correction(policy, obs, key):
    """Sample actions with proper tanh squashing and log prob correction."""
    # Get network output
    h = obs @ policy.w1.value + policy.b1.value
    if policy.use_layernorm:
        h = policy.layer_norm(h)
    h = jax.nn.relu(h)
    
    # Raw unbounded mean (before tanh)
    raw_mean = h @ policy.w2.value + policy.b2.value
    std = jnp.exp(policy.log_std.value)
    std = jnp.broadcast_to(std, raw_mean.shape)
    
    # Sample in unbounded space
    eps = jax.random.normal(key, raw_mean.shape)
    z = raw_mean + std * eps
    
    # Apply tanh squashing
    actions = MAX_TORQUE * jnp.tanh(z)
    
    # Log prob in unbounded space
    log_probs_z = gaussian_log_prob(z, raw_mean, std)
    
    # Correction for tanh squashing
    # For a = c * tanh(z), we have da/dz = c * (1 - tanh²(z))
    # log|da/dz| = log(c) + log(1 - tanh²(z))
    tanh_z = jnp.tanh(z)
    log_det_jacobian = jnp.log(MAX_TORQUE) + jnp.log(1.0 - tanh_z**2 + 1e-6)
    log_det_jacobian = jnp.sum(log_det_jacobian, axis=-1)  # Sum over action dims
    
    # Corrected log prob
    log_probs = log_probs_z - log_det_jacobian
    
    return actions, log_probs, raw_mean

# Test the difference
policy = GaussianPolicy(2, 1, hidden_dim=64)
obs = jnp.array([[0.0, 0.0], [jnp.pi, 0.0], [jnp.pi/2, 1.0]])
key = jax.random.PRNGKey(42)

# Current (incorrect) method
actions_old, log_probs_old = policy(obs), None
mean_old = actions_old[0]

# Correct method
actions_new, log_probs_new, raw_mean = sample_actions_with_tanh_correction(
    policy, obs, key
)

print("=== Tanh Squashing Analysis ===")
print(f"Observation shapes: {obs.shape}")
print("\nOld method (tanh on mean only):")
print(f"  Mean: {mean_old[:, 0]}")
print("  Actions would explode beyond [-2, 2] without clipping")

print("\nNew method (proper tanh squashing):")
print(f"  Raw mean: {raw_mean[:, 0]}")  
print(f"  Squashed mean: {MAX_TORQUE * jnp.tanh(raw_mean[:, 0])}")
print(f"  Sampled actions: {actions_new[:, 0]}")
print(f"  Log probs: {log_probs_new}")

print(f"\nActions are guaranteed in [{-MAX_TORQUE}, {MAX_TORQUE}]")