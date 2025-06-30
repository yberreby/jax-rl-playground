"""Running observation normalization."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple


class RunningMeanStd(NamedTuple):
    """Tracks running statistics for normalization."""
    mean: Float[Array, "obs_dim"]
    var: Float[Array, "obs_dim"]
    count: float
    
    
def init_normalizer(obs_dim: int) -> RunningMeanStd:
    """Initialize observation normalizer."""
    return RunningMeanStd(
        mean=jnp.zeros(obs_dim),
        var=jnp.ones(obs_dim),
        count=1e-4,
    )


@jax.jit
def update_normalizer(
    normalizer: RunningMeanStd,
    obs: Float[Array, "batch obs_dim"],
) -> RunningMeanStd:
    """Update running statistics with new observations."""
    batch_mean = jnp.mean(obs, axis=0)
    batch_var = jnp.var(obs, axis=0)
    batch_count = obs.shape[0]
    
    delta = batch_mean - normalizer.mean
    total_count = normalizer.count + batch_count
    
    new_mean = normalizer.mean + delta * batch_count / total_count
    m_a = normalizer.var * normalizer.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * normalizer.count * batch_count / total_count
    new_var = M2 / total_count
    
    return RunningMeanStd(
        mean=new_mean,
        var=new_var,
        count=total_count,
    )


@jax.jit
def normalize_obs(
    obs: Float[Array, "... obs_dim"],
    normalizer: RunningMeanStd,
    epsilon: float = 1e-8,
) -> Float[Array, "... obs_dim"]:
    """Normalize observations using running statistics."""
    return (obs - normalizer.mean) / jnp.sqrt(normalizer.var + epsilon)