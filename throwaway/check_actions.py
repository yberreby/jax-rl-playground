#!/usr/bin/env python3
"""Check what's happening with action bounds."""

import jax
import jax.numpy as jnp
from src.policy import GaussianPolicy, sample_actions
from src.pendulum import MAX_TORQUE

# Create policy
policy = GaussianPolicy(2, 1, hidden_dim=64)

# Test states
states = jnp.array([
    [0.0, 0.0],      # top
    [jnp.pi, 0.0],   # bottom
    [jnp.pi/2, 0.0], # side
    [-jnp.pi/2, 0.0], # other side
])

# Get means and stds
means, stds = policy(states)

print("=== Policy Output Analysis ===")
print(f"MAX_TORQUE = {MAX_TORQUE}")
print("\nDirect policy output (should be bounded by tanh):")
for i, (m, s) in enumerate(zip(means[:, 0], stds[:, 0])):
    print(f"State {i}: mean={m:6.3f}, std={s:6.3f}")

# Sample actions
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 10)

print("\n=== Sampled Actions ===")
all_actions = []
for k in keys:
    actions, log_probs = sample_actions(policy, states, k)
    all_actions.append(actions)
    
all_actions = jnp.stack(all_actions)
print(f"Shape: {all_actions.shape}")
print(f"Min: {all_actions.min():.3f}")
print(f"Max: {all_actions.max():.3f}")
print(f"Should be in [{-MAX_TORQUE}, {MAX_TORQUE}]")

# Check if tanh is actually being applied
print("\n=== Checking Tanh Application ===")
# Manually compute what should happen
h = states @ policy.w1.value + policy.b1.value
if policy.use_layernorm:
    h = policy.layer_norm(h)
h = jax.nn.relu(h)
raw_mean = h @ policy.w2.value + policy.b2.value
tanh_mean = MAX_TORQUE * jnp.tanh(raw_mean)

print("Raw mean (before tanh):", raw_mean[:, 0])
print("Tanh mean (should match policy output):", tanh_mean[:, 0])
print("Policy output:", means[:, 0])
print("Match?", jnp.allclose(tanh_mean, means))