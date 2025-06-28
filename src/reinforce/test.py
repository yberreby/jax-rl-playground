import jax
import jax.numpy as jnp
from src.reinforce import reinforce_loss


@jax.jit
def dummy_policy(params, obs):
    # Simple linear policy for testing
    mean = obs @ params["W"] + params["b"]
    std = jnp.ones_like(mean) * 0.1
    return mean, std


def test_reinforce_loss_basic():
    key = jax.random.PRNGKey(0)
    obs = jnp.ones((4, 3))
    actions = jnp.zeros((4, 2))
    rewards = jnp.ones(4)

    params = {"W": jax.random.normal(key, (3, 2)) * 0.1, "b": jnp.zeros(2)}

    loss = reinforce_loss(dummy_policy, params, obs, actions, rewards)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_reinforce_zero_rewards():
    params = {"W": jnp.ones((3, 2)), "b": jnp.zeros(2)}
    obs = jnp.ones((4, 3))
    actions = jnp.zeros((4, 2))
    rewards = jnp.zeros(4)

    loss = reinforce_loss(dummy_policy, params, obs, actions, rewards)
    assert loss == 0.0


def test_reinforce_gradient_flow():
    params = {"W": jnp.ones((3, 2)) * 0.1, "b": jnp.zeros(2)}
    obs = jnp.ones((10, 3))
    actions = jnp.ones((10, 2))
    rewards = jnp.ones(10)

    loss, grads = jax.value_and_grad(reinforce_loss, argnums=1)(
        dummy_policy, params, obs, actions, rewards
    )

    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))
