import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from src.policy import policy_forward


@jax.jit
def reinforce_loss(
    params: tuple[
        Float[Array, "in_dim hidden"],
        Float[Array, "hidden"],
        Float[Array, "hidden out_dim"],
        Float[Array, "out_dim"],
    ],
    obs: Float[Array, "batch in_dim"],
    actions: Float[Array, "batch out_dim"],
    rewards: Float[Array, "batch"],
) -> Float[Array, ""]:
    mean, std = policy_forward(params, obs)

    # log p(a|s) for Gaussian policy: -0.5 * [log(2π) + 2*log(σ) + ((a-μ)/σ)²]
    z_score = (actions - mean) / std
    log_normalizer = jnp.log(2 * jnp.pi) + 2 * jnp.log(std)
    log_probs = -0.5 * jnp.sum(z_score**2 + log_normalizer, axis=-1)

    # REINFORCE: maximize E[log π(a|s) * R]
    return -jnp.mean(log_probs * rewards)


def test_reinforce_loss():
    from src.policy import init_policy

    key = jax.random.PRNGKey(1)
    params = init_policy(key, obs_dim=8, action_dim=2)
    obs_batch = jnp.ones((4, 8))
    actions = jnp.zeros((4, 2))
    loss = reinforce_loss(params, obs_batch, actions, jnp.ones(4))
    assert loss.shape == ()
    assert jnp.isfinite(loss)
