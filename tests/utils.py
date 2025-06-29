import jax
import jax.numpy as jnp
from typing import Dict, Callable
from flax import nnx

# Standard gradient parameter names
GRADIENT_PARAMS = ["w1", "b1", "w2", "b2", "log_std"]


def supervised_loss_fn(
    policy: nnx.Module, obs: jax.Array, targets: jax.Array
) -> jax.Array:
    mean, _ = policy(obs)  # type: ignore[operator]
    return jnp.mean((mean - targets) ** 2)


def create_loss_fn(obs: jax.Array, targets: jax.Array) -> Callable:
    def loss_fn(policy):
        return supervised_loss_fn(policy, obs, targets)

    return loss_fn


def extract_gradient_norms(grads: nnx.Module) -> Dict[str, jax.Array]:
    grad_norms = {}
    for param in GRADIENT_PARAMS:
        grad_norms[param] = jnp.linalg.norm(getattr(grads, param).value)
    return grad_norms
