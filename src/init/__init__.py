import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from src.constants import DEFAULT_SPARSE_INIT_SPARSITY


def sparse_init(
    key: PRNGKeyArray, shape: tuple[int, ...], sparsity: float = DEFAULT_SPARSE_INIT_SPARSITY
) -> Float[Array, "..."]:
    fan_in = shape[-1] if len(shape) >= 2 else 1
    scale = 1.0 / jnp.sqrt(fan_in)

    mask_key, value_key = jax.random.split(key)
    mask = jax.random.uniform(mask_key, shape) > sparsity
    values = jax.random.normal(value_key, shape) * scale

    return values * mask
