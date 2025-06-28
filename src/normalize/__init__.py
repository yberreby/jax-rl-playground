import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from src.constants import DEFAULT_EPSILON, DEFAULT_CLIP_VALUE


@jax.jit
def normalize_obs(
    obs: Float[Array, "... obs_dim"],
    epsilon: float = DEFAULT_EPSILON,
    clip_value: float = DEFAULT_CLIP_VALUE,
) -> Float[Array, "... obs_dim"]:
    mean = jnp.mean(obs, axis=-1, keepdims=True)
    var = jnp.var(obs, axis=-1, keepdims=True)
    normalized = (obs - mean) / jnp.sqrt(var + epsilon)
    return jnp.clip(normalized, -clip_value, clip_value)


@jax.jit
def scale_rewards(
    rewards: Float[Array, "batch"],
    returns: Float[Array, "batch"],
    epsilon: float = DEFAULT_EPSILON,
) -> Float[Array, "batch"]:
    return_std = jnp.std(returns)
    return rewards / (return_std + epsilon)
