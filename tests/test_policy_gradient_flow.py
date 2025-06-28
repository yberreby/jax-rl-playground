import jax.numpy as jnp
import optax
from flax import nnx
from src.policy_nnx import GaussianPolicy
from tests.common_patterns import create_loss_fn


def test_policy_gradient_step():
    policy = GaussianPolicy(obs_dim=2, action_dim=1, hidden_dim=4)
    optimizer = nnx.Optimizer(policy, optax.adam(0.1))

    x = jnp.array([[1.0, 1.0]])
    target = jnp.array([[2.0]])

    # Get output before
    mean_before, _ = policy(x)

    # Take gradient step
    loss_fn = create_loss_fn(x, target)
    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(grads)

    # Get output after
    mean_after, _ = policy(x)

    print(f"Mean before: {mean_before[0, 0]:.4f}")
    print(f"Mean after: {mean_after[0, 0]:.4f}")
    print(f"Target: {target[0, 0]:.4f}")
    print(f"Loss: {loss:.4f}")

    # Should move toward target
    distance_before = jnp.abs(mean_before[0, 0] - target[0, 0])
    distance_after = jnp.abs(mean_after[0, 0] - target[0, 0])

    assert distance_after < distance_before, (
        f"Didn't move toward target: {distance_before:.4f} -> {distance_after:.4f}"
    )
