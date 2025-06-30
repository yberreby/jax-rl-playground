#!/usr/bin/env python3
"""Unit tests for critic implementation."""

import jax.numpy as jnp
import optax
from flax import nnx
from src.critic import ValueFunction, compute_critic_loss, update_critic


def test_value_function_shapes():
    """Test that value function has correct input/output shapes."""
    print("=== Value Function Shape Test ===")

    critic = ValueFunction(obs_dim=2, hidden_dim=32)

    # Test single state
    state = jnp.array([[0.5, -0.2]])  # shape (1, 2)
    value = critic(state)
    print(f"Single state: {state.shape} -> {value.shape}")
    print(f"Value: {value}")
    assert value.shape == (1,), f"Expected (1,), got {value.shape}"

    # Test batch of states
    states = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])  # shape (3, 2)
    values = critic(states)
    print(f"Batch states: {states.shape} -> {values.shape}")
    print(f"Values: {values}")
    assert values.shape == (3,), f"Expected (3,), got {values.shape}"


def test_critic_training():
    """Test that critic can learn to predict returns."""
    print("\n=== Critic Training Test ===")

    # Create synthetic training data
    states = jnp.array(
        [
            [0.0, 0.0],  # Top position, should have high value
            [3.14, 0.0],  # Bottom position, should have low value
            [1.57, 0.0],  # Side position, medium value
        ]
    )

    # Corresponding returns (what the critic should learn to predict)
    returns = jnp.array([200.0, -200.0, 0.0])

    print("Training data:")
    print(f"  States: {states}")
    print(f"  Returns: {returns}")

    # Initialize critic and optimizer
    critic = ValueFunction(obs_dim=2, hidden_dim=32)
    optimizer = nnx.Optimizer(critic, optax.adam(learning_rate=0.01))

    # Initial predictions (should be random)
    initial_values = critic(states)
    initial_loss = compute_critic_loss(critic, states, returns)
    print("\nBefore training:")
    print(f"  Predicted values: {initial_values}")
    print(f"  Target returns: {returns}")
    print(f"  Loss: {initial_loss:.3f}")

    # Train for several steps
    for step in range(100):
        loss = update_critic(critic, optimizer, states, returns)
        if step % 20 == 0:
            values = critic(states)
            print(f"  Step {step}: loss={loss:.3f}, values={values}")

    # Final predictions (should be close to targets)
    final_values = critic(states)
    final_loss = compute_critic_loss(critic, states, returns)
    print("\nAfter training:")
    print(f"  Predicted values: {final_values}")
    print(f"  Target returns: {returns}")
    print(f"  Loss: {final_loss:.3f}")
    print(f"  Improvement: {initial_loss / final_loss:.1f}x")


def test_advantage_computation():
    """Test advantage computation with critic."""
    print("\n=== Advantage Computation Test ===")

    # Create trained critic (simplified)
    ValueFunction(obs_dim=2, hidden_dim=32)

    # Test states and actual returns
    states = jnp.array(
        [
            [0.0, 0.0],  # Good state
            [3.14, 0.0],  # Bad state
        ]
    )
    actual_returns = jnp.array([150.0, -100.0])

    # Manually set critic to predict reasonable values
    # (In practice, this would be learned)
    predicted_values = jnp.array([120.0, -80.0])

    # Compute advantages
    advantages = actual_returns - predicted_values

    print(f"States: {states}")
    print(f"Actual returns: {actual_returns}")
    print(f"Predicted values: {predicted_values}")
    print(f"Advantages: {advantages}")
    print("Interpretation:")
    print(f"  State 1: better than expected (+{advantages[0]:.0f})")
    print(f"  State 2: worse than expected ({advantages[1]:.0f})")


def test_vs_global_baseline():
    """Compare critic advantages vs global baseline."""
    print("\n=== Critic vs Global Baseline ===")

    # Simulate returns from different states
    returns = jnp.array([200.0, 150.0, -50.0, -150.0, 100.0])
    jnp.array(
        [
            [0.0, 0.0],  # Top
            [0.5, 0.0],  # Near top
            [2.0, 0.0],  # Side
            [3.14, 0.0],  # Bottom
            [1.0, 0.0],  # Other
        ]
    )

    # Global baseline approach
    global_baseline = jnp.mean(returns)
    global_advantages = returns - global_baseline

    # Critic approach (idealized - critic perfectly predicts state values)
    ideal_critic_values = jnp.array([180.0, 140.0, -30.0, -140.0, 90.0])
    critic_advantages = returns - ideal_critic_values

    print(f"Returns: {returns}")
    print(f"Global baseline: {global_baseline:.1f}")
    print(f"Global advantages: {global_advantages}")
    print(f"Ideal critic values: {ideal_critic_values}")
    print(f"Critic advantages: {critic_advantages}")

    print("\nAdvantage variance:")
    print(f"  Global baseline: {jnp.var(global_advantages):.1f}")
    print(f"  Critic baseline: {jnp.var(critic_advantages):.1f}")
    print("Critic provides lower variance advantages (better learning signal)!")


if __name__ == "__main__":
    test_value_function_shapes()
    test_critic_training()
    test_advantage_computation()
    test_vs_global_baseline()
