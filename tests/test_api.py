import jax
import jax.numpy as jnp
import optax
import pytest
import src


def test_full_pipeline():
    key = jax.random.PRNGKey(42)

    params = src.init_policy(key, obs_dim=8, action_dim=2)

    obs = jax.random.normal(key, (32, 8))
    mean, std = src.policy_forward(params, obs)

    key, action_key = jax.random.split(key)
    actions, log_probs = src.sample_actions(action_key, mean, std)

    rewards = jax.random.normal(key, (32,))
    returns = jnp.cumsum(rewards)
    scaled_rewards = src.scale_rewards(rewards, returns)

    loss = src.reinforce_loss(params, obs, actions, scaled_rewards)

    assert jnp.isfinite(loss)
    assert loss.shape == ()


@pytest.mark.slow
def test_learning_with_public_api():
    key = jax.random.PRNGKey(0)
    params = src.init_policy(key, obs_dim=4, action_dim=1)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    target_mean = 2.0
    obs = jnp.ones((100, 4))

    losses = []
    for i in range(200):
        mean, std = src.policy_forward(params, obs)
        actions = jnp.full((100, 1), target_mean)

        advantage = target_mean - mean.mean()
        rewards = jnp.full(100, advantage)

        loss, grads = jax.value_and_grad(src.reinforce_loss)(
            params, obs, actions, rewards
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        losses.append(float(loss))

    final_mean, _ = src.policy_forward(params, obs[:1])
    assert abs(float(final_mean[0, 0]) - target_mean) < 1.0
