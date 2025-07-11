import jax.numpy as jnp
import optax
from flax import nnx
from tests.fixtures import create_test_policy, create_test_data
from tests.constants import DEFAULT_LEARNING_RATE
from tests.utils import create_loss_fn


def test_gradients_exist():
    policy = create_test_policy()
    _, obs, targets = create_test_data(batch_size=1)

    loss_fn = create_loss_fn(obs, targets)

    _, grads = nnx.value_and_grad(loss_fn)(policy)

    # Try to access gradients
    assert jnp.any(grads.w1 != 0)


def test_parameters_update():
    policy = create_test_policy()
    optimizer = nnx.Optimizer(policy, optax.sgd(DEFAULT_LEARNING_RATE))

    w1_before = policy.w1.value.copy()

    _, obs, targets = create_test_data(batch_size=1)
    loss_fn = create_loss_fn(obs, targets)

    _, grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(grads)

    assert not jnp.allclose(w1_before, policy.w1.value)
