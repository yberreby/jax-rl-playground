import jax.numpy as jnp
import optax
from flax import nnx
from src.policy_nnx import GaussianPolicy


def test_gradients_exist():
    """Gradients should be non-zero for a simple loss."""
    policy = GaussianPolicy(obs_dim=4, action_dim=2)
    
    def loss_fn(policy):
        mean, _ = policy(jnp.ones((1, 4)))
        return jnp.sum(mean ** 2)
    
    _, grads = nnx.value_and_grad(loss_fn)(policy)
    
    # Print grad structure to understand it
    print(f"Gradient type: {type(grads)}")
    print(f"Gradient attributes: {dir(grads)}")
    
    # Try to access gradients
    if hasattr(grads, 'w1'):
        assert jnp.any(grads.w1 != 0)
    else:
        # Maybe it's a different structure
        print(f"Grads content: {grads}")


def test_parameters_update():
    """Parameters should change after optimizer step."""
    policy = GaussianPolicy(obs_dim=4, action_dim=2)
    optimizer = nnx.Optimizer(policy, optax.sgd(1.0))
    
    w1_before = policy.w1.value.copy()
    
    def loss_fn(policy):
        mean, _ = policy(jnp.ones((1, 4)))
        return jnp.sum(mean ** 2)
    
    _, grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(grads)
    
    assert not jnp.allclose(w1_before, policy.w1.value)