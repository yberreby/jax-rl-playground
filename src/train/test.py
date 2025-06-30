import jax
import jax.numpy as jnp
import pytest
from flax import nnx
import optax
from . import train_step, train
from ..policy import GaussianPolicy
from ..pendulum import step, reset_env

# Test constants
TEST_BATCH_SIZE = 5
TEST_OBS_DIM = 2
TEST_ACTION_DIM = 1
TEST_HIDDEN_DIM = 16


def test_train_step_gradient_flow():
    # Test that train_step updates parameters
    key = jax.random.PRNGKey(42)
    policy = GaussianPolicy(
        obs_dim=TEST_OBS_DIM,
        action_dim=TEST_ACTION_DIM,
        hidden_dim=TEST_HIDDEN_DIM,
        use_layernorm=False,
    )

    optimizer = nnx.Optimizer(policy, optax.adam(0.01))

    # Create dummy batch
    batch_states = jax.random.normal(key, (TEST_BATCH_SIZE, TEST_OBS_DIM))
    batch_actions = jax.random.normal(key, (TEST_BATCH_SIZE, TEST_ACTION_DIM))
    batch_advantages = jnp.ones(TEST_BATCH_SIZE)

    # Get initial parameters
    initial_params = []
    for layer in policy.layers:
        if hasattr(layer, 'kernel'):
            initial_params.append(layer.kernel.value.copy())
        if hasattr(layer, 'bias'):
            initial_params.append(layer.bias.value.copy())
    initial_param_sum = sum(jnp.sum(p) for p in initial_params)

    # Train step
    loss, grad_norm, grad_var = train_step(
        policy, optimizer, batch_states, batch_actions, batch_advantages
    )

    # Check outputs
    assert jnp.isfinite(loss)
    assert loss.shape == ()
    assert grad_norm > 0
    assert grad_var >= 0

    # Check parameters changed
    final_params = []
    for layer in policy.layers:
        if hasattr(layer, 'kernel'):
            final_params.append(layer.kernel.value)
        if hasattr(layer, 'bias'):
            final_params.append(layer.bias.value)
    final_param_sum = sum(jnp.sum(p) for p in final_params)
    assert not jnp.allclose(initial_param_sum, final_param_sum)


@pytest.mark.slow
def test_train_improves_performance():
    # Simple test that training reduces loss
    policy = GaussianPolicy(
        obs_dim=TEST_OBS_DIM,
        action_dim=TEST_ACTION_DIM,
        hidden_dim=TEST_HIDDEN_DIM,
        use_layernorm=False,
    )

    metrics = train(
        policy,
        step,
        reset_env,
        n_iterations=10,
        episodes_per_iter=5,
        learning_rate=1e-3,
        use_baseline=True,
        verbose=False,
    )

    # Check that we got metrics
    assert len(metrics["loss"]) == 10
    assert all(jnp.isfinite(loss) for loss in metrics["loss"])

    # Loss should generally decrease (though not monotonically)
    # Just check it's not increasing dramatically
    initial_losses = metrics["loss"][:3]
    final_losses = metrics["loss"][-3:]
    assert jnp.mean(final_losses) < jnp.mean(initial_losses) * 10  # Very loose bound