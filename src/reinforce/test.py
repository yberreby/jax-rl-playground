import jax
import jax.numpy as jnp
from src.reinforce import reinforce_loss
from src.policy import init_policy


def test_reinforce_gradient_flow():
    key = jax.random.PRNGKey(0)
    params = init_policy(key, obs_dim=4, action_dim=2)

    obs = jax.random.normal(key, (32, 4))
    actions = jax.random.normal(key, (32, 2))
    rewards = jax.random.normal(key, (32,))

    loss, grads = jax.value_and_grad(reinforce_loss)(params, obs, actions, rewards)

    assert jnp.isfinite(loss)
    for grad in grads:
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0)


def test_reinforce_zero_rewards():
    key = jax.random.PRNGKey(0)
    params = init_policy(key, obs_dim=4, action_dim=2)

    obs = jax.random.normal(key, (10, 4))
    actions = jax.random.normal(key, (10, 2))
    rewards = jnp.zeros(10)

    loss = reinforce_loss(params, obs, actions, rewards)
    assert loss == 0.0
