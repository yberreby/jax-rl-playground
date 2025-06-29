import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple


class BaselineState(NamedTuple):
    mean: Float[Array, ""]
    n_samples: int  # Renamed from count to avoid conflict


@jax.jit
def update_baseline(
    state: BaselineState, returns: Float[Array, "batch"]
) -> BaselineState:
    batch_mean = jnp.mean(returns)
    batch_size = returns.shape[0]

    # Incremental mean update
    total_count = state.n_samples + batch_size
    new_mean = (state.mean * state.n_samples + batch_mean * batch_size) / total_count

    return BaselineState(mean=new_mean, n_samples=total_count)


@jax.jit
def compute_advantages(
    returns: Float[Array, "batch"], baseline: Float[Array, ""]
) -> Float[Array, "batch"]:
    return returns - baseline


def init_baseline() -> BaselineState:
    return BaselineState(mean=jnp.array(0.0), n_samples=0)
