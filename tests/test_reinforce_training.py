import jax
import jax.numpy as jnp
import optax
import pytest

from src.policy import init_policy, policy_forward
from src.reinforce import reinforce_loss
from src.normalize import scale_rewards


@pytest.mark.slow
def test_learning_dynamics():
    key = jax.random.PRNGKey(0)
    params = init_policy(key, obs_dim=4, action_dim=1)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    target_mean = 2.0
    obs = jnp.ones((100, 4))

    losses = []
    for i in range(200):
        mean, std = policy_forward(params, obs)
        actions = jnp.full((100, 1), target_mean)

        advantage = target_mean - mean.mean()
        rewards = jnp.full(100, advantage)

        loss, grads = jax.value_and_grad(reinforce_loss)(params, obs, actions, rewards)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        losses.append(float(loss))

    final_mean, _ = policy_forward(params, obs[:1])
    assert abs(float(final_mean[0, 0]) - target_mean) < 1.0


def test_full_pipeline():
    key = jax.random.PRNGKey(42)

    params = init_policy(key, obs_dim=8, action_dim=2)

    obs = jax.random.normal(key, (32, 8))
    mean, std = policy_forward(params, obs)

    from src.policy import sample_actions

    key, action_key = jax.random.split(key)
    actions, log_probs = sample_actions(action_key, mean, std)

    rewards = jax.random.normal(key, (32,))
    returns = jnp.cumsum(rewards)
    scaled_rewards = scale_rewards(rewards, returns)

    loss = reinforce_loss(params, obs, actions, scaled_rewards)

    assert jnp.isfinite(loss)
    assert loss.shape == ()
