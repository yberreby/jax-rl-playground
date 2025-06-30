"""Value function (critic) implementation."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from flax import nnx


class ValueFunction(nnx.Module):
    """Neural network value function V(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)

        key1, key2, key3 = jax.random.split(rngs(), 3)

        self.layers = nnx.Sequential(
            nnx.Linear(obs_dim, hidden_dim, rngs=nnx.Rngs(key1)),
            nnx.tanh,
            nnx.Linear(hidden_dim, hidden_dim, rngs=nnx.Rngs(key2)),
            nnx.tanh,
            nnx.Linear(hidden_dim, 1, rngs=nnx.Rngs(key3)),  # Single output
        )

    def __call__(self, obs: Float[Array, "batch obs_dim"]) -> Float[Array, "batch"]:
        """Predict value V(s) for each state."""
        values = self.layers(obs)
        return values.squeeze(-1)  # Remove last dimension


@nnx.jit
def compute_critic_loss(
    critic: ValueFunction,
    states: Float[Array, "batch obs_dim"],
    returns: Float[Array, "batch"],
) -> Float[Array, ""]:
    """Compute critic loss: MSE between V(s) and actual returns."""
    predicted_values = critic(states)
    loss = jnp.mean((predicted_values - returns) ** 2)
    return loss


@nnx.jit
def update_critic(
    critic: ValueFunction,
    optimizer: nnx.Optimizer,
    states: Float[Array, "batch obs_dim"],
    returns: Float[Array, "batch"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Update critic parameters and return metrics."""

    def loss_fn(critic):
        predicted_values = critic(states)
        return jnp.mean((predicted_values - returns) ** 2), predicted_values

    (loss, predicted_values), grads = nnx.value_and_grad(loss_fn, has_aux=True)(critic)
    optimizer.update(grads)
    
    # Compute gradient norm
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    
    # Compute prediction statistics
    mean_pred = jnp.mean(predicted_values)
    std_pred = jnp.std(predicted_values)
    
    return loss, mean_pred, std_pred, grad_norm


def compute_critic_advantages(
    critic: ValueFunction,
    states: Float[Array, "batch obs_dim"],
    returns: Float[Array, "batch"],
) -> Float[Array, "batch"]:
    """Compute advantages using critic as baseline."""
    predicted_values = critic(states)
    advantages = returns - predicted_values
    return advantages
