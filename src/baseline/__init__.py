import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple


class BaselineState(NamedTuple):
    """Running mean baseline for variance reduction."""
    mean: Float[Array, ""]
    n_samples: int


@jax.jit
def update_baseline(
    state: BaselineState, values: Float[Array, "batch"]
) -> BaselineState:
    """Update running mean with new values."""
    batch_size = values.shape[0]
    batch_mean = jnp.mean(values)
    
    # Running average
    total_count = state.n_samples + batch_size
    new_mean = (state.mean * state.n_samples + batch_mean * batch_size) / total_count
    
    return BaselineState(mean=new_mean, n_samples=total_count)


@jax.jit
def compute_advantages(
    returns: Float[Array, "batch"], baseline: Float[Array, ""]
) -> Float[Array, "batch"]:
    """Compute advantages as returns - baseline."""
    return returns - baseline


def init_baseline() -> BaselineState:
    """Initialize baseline with zero mean."""
    return BaselineState(mean=jnp.array(0.0), n_samples=0)