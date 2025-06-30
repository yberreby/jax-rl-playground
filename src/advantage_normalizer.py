"""Simple advantage normalization."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


@jax.jit
def normalize_advantages(
    advantages: Float[Array, "batch"],
    epsilon: float = 1e-8,
) -> Float[Array, "batch"]:
    """Normalize advantages to have zero mean and unit variance.
    
    Standard trick for stable policy gradient training.
    """
    mean = jnp.mean(advantages)
    std = jnp.std(advantages)
    return (advantages - mean) / (std + epsilon)