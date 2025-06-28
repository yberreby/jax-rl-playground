import jax
import jax.numpy as jnp
import optax
import pytest
import matplotlib.pyplot as plt
import pandas as pd
from flax import nnx
from src.policy_nnx import GaussianPolicy


@pytest.mark.slow
def test_policy_learns_target_mean():
    """Test that a policy can learn to output a target mean action."""
    # Setup
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(obs_dim=4, action_dim=2, rngs=nnx.Rngs(key))
    optimizer = nnx.Optimizer(policy, optax.adam(3e-2))

    # Fixed input and target
    obs = jnp.ones((1, 4))
    target_mean = jnp.array([[2.0, -1.0]])

    # Training
    losses = []
    means = []

    @nnx.jit
    def train_step(policy, optimizer, obs, target):
        def loss_fn(policy):
            mean, _ = policy(obs)
            return jnp.mean((mean - target) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(policy)
        optimizer.update(grads)
        return loss

    # Train for 500 steps
    for step in range(500):
        loss = train_step(policy, optimizer, obs, target_mean)
        mean, _ = policy(obs)

        losses.append(float(loss))
        means.append(mean[0].tolist())

    # Save results
    df = pd.DataFrame(
        {
            "step": range(500),
            "loss": losses,
            "mean_0": [m[0] for m in means],
            "mean_1": [m[1] for m in means],
        }
    )
    df.to_csv("tests/outputs/policy_learning.csv", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Loss During Training")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.plot([m[0] for m in means], label="Output dim 0")
    ax2.plot([m[1] for m in means], label="Output dim 1")
    ax2.axhline(y=2.0, color="r", linestyle="--", alpha=0.5, label="Target dim 0")
    ax2.axhline(y=-1.0, color="b", linestyle="--", alpha=0.5, label="Target dim 1")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Action")
    ax2.set_title("Policy Output Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tests/outputs/policy_learning.png", dpi=150)
    plt.close()

    # Check convergence
    final_mean, _ = policy(obs)
    error = jnp.sqrt(jnp.sum((final_mean - target_mean) ** 2))
    assert error < 0.5, f"Did not converge. Error: {error:.3f}"
