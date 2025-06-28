import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.jit
def normalize_obs(
    obs: Float[Array, "... obs_dim"], epsilon: float = 1e-8, clip_value: float = 5.0
) -> Float[Array, "... obs_dim"]:
    mean = jnp.mean(obs, axis=-1, keepdims=True)
    var = jnp.var(obs, axis=-1, keepdims=True)
    normalized = (obs - mean) / jnp.sqrt(var + epsilon)
    return jnp.clip(normalized, -clip_value, clip_value)


@jax.jit
def scale_rewards(
    rewards: Float[Array, "batch"],
    returns: Float[Array, "batch"],
    epsilon: float = 1e-8,
) -> Float[Array, "batch"]:
    return_std = jnp.std(returns)
    return rewards / (return_std + epsilon)
