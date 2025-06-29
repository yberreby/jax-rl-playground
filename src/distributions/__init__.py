import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# TODO: use standard APIs
@jax.jit
def gaussian_log_prob(
    x: Float[Array, "... dim"],
    mean: Float[Array, "... dim"],
    std: Float[Array, "... dim"],
) -> Float[Array, "..."]:
    z_score = (x - mean) / std
    log_normalizer = jnp.log(2 * jnp.pi) + 2 * jnp.log(std)
    return -0.5 * jnp.sum(z_score**2 + log_normalizer, axis=-1)
