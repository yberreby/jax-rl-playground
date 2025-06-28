import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def sparse_init(
    key: PRNGKeyArray, shape: tuple[int, ...], sparsity: float = 0.9
) -> Float[Array, "..."]:
    fan_in = shape[-1] if len(shape) >= 2 else 1
    scale = 1.0 / jnp.sqrt(fan_in)

    mask_key, value_key = jax.random.split(key)
    mask = jax.random.uniform(mask_key, shape) > sparsity
    values = jax.random.normal(value_key, shape) * scale

    return values * mask


def test_sparse_init():
    key = jax.random.PRNGKey(0)
    W = sparse_init(key, (100, 50), sparsity=0.9)
    assert W.shape == (100, 50)
    nonzero_frac = jnp.mean(W != 0)
    assert 0.05 < nonzero_frac < 0.15
