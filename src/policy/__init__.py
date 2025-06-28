import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from src.normalize import normalize_obs
from src.init import sparse_init


@jax.jit
def policy_forward(
    params: tuple[
        Float[Array, "in_dim hidden"],
        Float[Array, "hidden"],
        Float[Array, "hidden out_dim"],
        Float[Array, "out_dim"],
    ],
    obs: Float[Array, "batch in_dim"],
) -> tuple[Float[Array, "batch out_dim"], Float[Array, "batch out_dim"]]:
    W1, b1, W2, b2 = params

    h = obs @ W1 + b1
    h = jax.nn.relu(h)
    h = normalize_obs(h)

    mean = h @ W2 + b2
    log_std = jnp.zeros_like(mean) - 1.0

    return mean, jnp.exp(log_std)


def test_policy_forward():
    obs_batch = jnp.ones((4, 8))
    W1 = jnp.ones((8, 32)) * 0.1
    b1 = jnp.zeros(32)
    W2 = jnp.ones((32, 2)) * 0.1
    b2 = jnp.zeros(2)
    params = (W1, b1, W2, b2)
    mean, std = policy_forward(params, obs_batch)
    assert mean.shape == (4, 2)
    assert std.shape == (4, 2)


def init_policy(
    key: PRNGKeyArray,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    sparsity: float = 0.9,
) -> tuple[
    Float[Array, "obs_dim hidden"],
    Float[Array, "hidden"],
    Float[Array, "hidden action_dim"],
    Float[Array, "action_dim"],
]:
    k1, k2 = jax.random.split(key)
    W1 = sparse_init(k1, (obs_dim, hidden_dim), sparsity)
    b1 = jnp.zeros(hidden_dim)
    W2 = sparse_init(k2, (hidden_dim, action_dim), sparsity)
    b2 = jnp.zeros(action_dim)
    return (W1, b1, W2, b2)


def test_init_policy():
    params = init_policy(jax.random.PRNGKey(1), obs_dim=8, action_dim=2)
    assert len(params) == 4
    assert params[0].shape == (8, 64)
    assert params[2].shape == (64, 2)


@jax.jit
def sample_actions(
    key: PRNGKeyArray,
    mean: Float[Array, "batch action_dim"],
    std: Float[Array, "batch action_dim"],
) -> tuple[Float[Array, "batch action_dim"], Float[Array, "batch"]]:
    eps = jax.random.normal(key, mean.shape)
    actions = mean + std * eps

    log_probs = -0.5 * jnp.sum(
        jnp.square((actions - mean) / std) + 2 * jnp.log(std) + jnp.log(2 * jnp.pi),
        axis=-1,
    )

    return actions, log_probs


def test_sample_actions():
    key = jax.random.PRNGKey(2)
    mean = jnp.zeros((4, 2))
    std = jnp.ones((4, 2))
    actions, log_probs = sample_actions(key, mean, std)
    assert actions.shape == (4, 2)
    assert log_probs.shape == (4,)
