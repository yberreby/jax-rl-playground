import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike
from functools import partial
from typing import Optional


@partial(jax.jit, static_argnames=["window_size"])
def exponential_moving_average(
    values: Float[Array, "batch time features"],
    alpha: ScalarLike,
    window_size: Optional[int] = None,
) -> Float[Array, "batch time features"]:
    if window_size is not None:
        alpha = 2 / (window_size + 1)

    def ema_step(carry, x):
        avg = carry
        avg = alpha * x + (1 - alpha) * avg
        return avg, avg

    _, ema_values = jax.lax.scan(ema_step, values[:, 0], values.swapaxes(0, 1))
    return ema_values.swapaxes(0, 1)


@jax.jit
def batch_correlate(
    x: Float[Array, "batch n"], y: Float[Array, "batch n"]
) -> Float[Array, "batch (2*n-1)"]:
    # Use jnp.correlate for proper cross-correlation
    def correlate_pair(xy):
        return jnp.correlate(xy[0], xy[1], mode="full")

    return jax.vmap(correlate_pair)((x, y))
