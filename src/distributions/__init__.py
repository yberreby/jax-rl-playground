import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.jit
def gaussian_log_prob(
    x: Float[Array, "... dim"],
    mean: Float[Array, "... dim"],
    std: Float[Array, "... dim"],
) -> Float[Array, "..."]:
    """Log probability of x under Gaussian(mean, std^2)."""
    z_score = (x - mean) / std
    log_normalizer = jnp.log(2 * jnp.pi) + 2 * jnp.log(std)
    return -0.5 * jnp.sum(z_score**2 + log_normalizer, axis=-1)