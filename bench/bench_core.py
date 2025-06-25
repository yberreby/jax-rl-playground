"""Benchmark JAX functions using richbench."""

import os

# Force CPU before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from src.core import batch_correlate, exponential_moving_average


def ema_small():
    """EMA on small batch (10x100x5)."""
    x = jnp.ones((10, 100, 5))
    exponential_moving_average(x, 0.9).block_until_ready()


def ema_medium():
    """EMA on medium batch (50x500x10)."""
    x = jnp.ones((50, 500, 10))
    exponential_moving_average(x, 0.9).block_until_ready()


def ema_large():
    """EMA on large batch (100x1000x20)."""
    x = jnp.ones((100, 1000, 20))
    exponential_moving_average(x, 0.9).block_until_ready()


def correlate_small():
    """Batch correlate small signals."""
    key = jax.random.PRNGKey(0)
    signals = jax.random.normal(key, (10, 100))
    kernels = jax.random.normal(key, (10, 100))
    batch_correlate(signals, kernels).block_until_ready()


def correlate_large():
    """Batch correlate large signals."""
    key = jax.random.PRNGKey(0)
    signals = jax.random.normal(key, (100, 1000))
    kernels = jax.random.normal(key, (100, 1000))
    batch_correlate(signals, kernels).block_until_ready()


def ema_with_jit():
    """EMA with JIT compilation."""
    x = jnp.ones((50, 500, 10))
    exponential_moving_average(x, 0.9).block_until_ready()


def ema_without_jit():
    """EMA without JIT (baseline)."""
    x = jnp.ones((50, 500, 10))
    # Use the non-jitted underlying function
    ema_fn = exponential_moving_average.__wrapped__
    ema_fn(x, 0.9).block_until_ready()


# Define benchmarks as A vs B comparisons
__benchmarks__ = [
    (ema_medium, ema_small, "Small vs medium batch size"),
    (ema_large, ema_medium, "Medium vs large batch size"),
    (correlate_large, correlate_small, "Small vs large correlation"),
    (ema_without_jit, ema_with_jit, "JIT vs no JIT"),
]
