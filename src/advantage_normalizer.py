"""Stable advantage normalization using running statistics."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import NamedTuple


class RunningStats(NamedTuple):
    """Running statistics for advantage normalization."""
    mean: Float[Array, ""]
    var: Float[Array, ""]
    count: Float[Array, ""]


@jax.jit
def update_running_stats(
    stats: RunningStats, 
    values: Float[Array, "batch"]
) -> RunningStats:
    """Update running statistics using Welford's algorithm."""
    batch_size = values.shape[0]
    batch_mean = jnp.mean(values)
    batch_var = jnp.var(values)
    
    # Welford's update
    new_count = stats.count + batch_size
    delta = batch_mean - stats.mean
    new_mean = stats.mean + delta * batch_size / new_count
    
    # Update variance (simplified version)
    # For full Welford's, we'd need sum of squares
    # Here we use exponential moving average as approximation
    alpha = jnp.minimum(batch_size / new_count, 0.1)  # Adaptive learning rate
    new_var = (1 - alpha) * stats.var + alpha * batch_var
    
    return RunningStats(mean=new_mean, var=new_var, count=new_count)


@jax.jit
def normalize_advantages(
    advantages: Float[Array, "batch"],
    stats: RunningStats,
    min_std: float = 1.0,
) -> Float[Array, "batch"]:
    """Normalize advantages using running statistics.
    
    Args:
        advantages: Raw advantages
        stats: Running statistics
        min_std: Minimum std to prevent division issues
        
    Returns:
        Normalized advantages (scaled but not centered)
    """
    running_std = jnp.sqrt(stats.var + 1e-8)
    running_std = jnp.maximum(running_std, min_std)
    
    # Just scale, don't center (as per user's request)
    normalized = advantages / running_std
    
    return normalized


def init_running_stats() -> RunningStats:
    """Initialize running statistics."""
    return RunningStats(
        mean=jnp.array(0.0),
        var=jnp.array(1.0),  # Start with unit variance
        count=jnp.array(0.0),
    )