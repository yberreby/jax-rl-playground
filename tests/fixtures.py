import jax
import jax.numpy as jnp
from flax import nnx
import optax
from src.policy_nnx import GaussianPolicy
from tests.constants import (
    DEFAULT_SEED,
    DEFAULT_OBS_DIM,
    DEFAULT_ACTION_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
)


def create_test_policy(
    obs_dim: int = DEFAULT_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    use_layernorm: bool = True,
    seed: int = DEFAULT_SEED,
) -> GaussianPolicy:
    rngs = nnx.Rngs(seed)
    return GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        use_layernorm=use_layernorm,
        rngs=rngs,
    )


def create_test_data(
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    obs_dim: int = DEFAULT_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    target_type: str = "fixed",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    key = jax.random.PRNGKey(seed)
    obs_key, target_key = jax.random.split(key)

    # Standard Gaussian observations
    obs = jax.random.normal(obs_key, (batch_size, obs_dim))

    # Different target types
    if target_type == "fixed":
        target_values = jnp.array([[1.0, -0.5]])
        targets = jnp.broadcast_to(target_values, (batch_size, action_dim))
    elif target_type == "bimodal":
        # Half at one mode, half at another
        target1 = jnp.array([[2.0, -1.0]])
        target2 = jnp.array([[-2.0, 1.0]])
        targets = jnp.concatenate(
            [
                jnp.tile(target1, (batch_size // 2, 1)),
                jnp.tile(target2, (batch_size // 2, 1)),
            ]
        )
    elif target_type == "random":
        targets = jax.random.normal(target_key, (batch_size, action_dim))
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return key, obs, targets


def create_optimizer(
    policy: nnx.Module,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    optimizer_name: str = "adam",
) -> nnx.Optimizer:
    if optimizer_name == "adam":
        opt = optax.adam(learning_rate)
    elif optimizer_name == "sgd":
        opt = optax.sgd(learning_rate)
    elif optimizer_name == "rmsprop":
        opt = optax.rmsprop(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return nnx.Optimizer(policy, opt)


def create_non_learnable_layernorm(num_features: int, rngs: nnx.Rngs) -> nnx.LayerNorm:
    """Create LayerNorm without learnable parameters (as per Elsayed et al.)"""
    return nnx.LayerNorm(
        num_features=num_features, use_bias=False, use_scale=False, rngs=rngs
    )
