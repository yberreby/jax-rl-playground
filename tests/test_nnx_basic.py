import jax
import jax.numpy as jnp
import optax
from flax import nnx


def test_basic_nnx_training():
    class Linear(nnx.Module):
        def __init__(self, din, dout, rngs):
            self.w = nnx.Param(jax.random.normal(rngs.params(), (din, dout)) * 0.1)
            self.b = nnx.Param(jnp.zeros((dout,)))

        def __call__(self, x):
            return x @ self.w + self.b

    # Create model
    model = Linear(2, 1, nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(0.1))

    # Simple data
    x = jnp.array([[1.0, 2.0]])
    y_target = jnp.array([[3.0]])

    # Get initial output
    y_before = model(x)

    # One gradient step
    def loss_fn(model):
        y_pred = model(x)
        return jnp.mean((y_pred - y_target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    # Check output changed
    y_after = model(x)

    print(f"Before: {y_before[0, 0]:.4f}")
    print(f"After: {y_after[0, 0]:.4f}")
    print(f"Target: {y_target[0, 0]:.4f}")
    print(f"Loss: {loss:.4f}")

    assert y_before != y_after, "Model output didn't change!"
