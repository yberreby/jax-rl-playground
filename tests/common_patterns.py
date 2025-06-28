import jax
import jax.numpy as jnp
from typing import Dict, Callable, Tuple, TYPE_CHECKING
from flax import nnx

if TYPE_CHECKING:
    pass

# Standard gradient parameter names
GRADIENT_PARAMS = ['w1', 'b1', 'w2', 'b2', 'log_std']


def supervised_loss_fn(
    policy: nnx.Module, 
    obs: jax.Array, 
    targets: jax.Array
) -> jax.Array:
    mean, _ = policy(obs)  # type: ignore[operator]
    return jnp.mean((mean - targets) ** 2)


def create_loss_fn(obs: jax.Array, targets: jax.Array) -> Callable:
    def loss_fn(policy):
        return supervised_loss_fn(policy, obs, targets)
    return loss_fn


def extract_gradient_norms(grads: nnx.Module) -> Dict[str, jax.Array]:
    grad_norms = {}
    for param in GRADIENT_PARAMS:
        if hasattr(grads, param):
            grad_norms[param] = jnp.linalg.norm(getattr(grads, param).value)
    return grad_norms


def compute_gradient_step(
    policy: nnx.Module,
    obs: jax.Array,
    targets: jax.Array
) -> Tuple[jax.Array, Dict[str, jax.Array], nnx.Module]:
    loss_fn = create_loss_fn(obs, targets)
    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    grad_norms = extract_gradient_norms(grads)
    return loss, grad_norms, grads


def print_metrics(title: str, metrics: Dict[str, float], indent: str = "  "):
    print(f"\n{title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{indent}{key}: {value:.6f}")
        else:
            print(f"{indent}{key}: {value}")


def print_gradient_info(grad_norms: Dict[str, float], prefix: str = ""):
    if prefix:
        print(f"\n{prefix} Gradient Norms:")
    else:
        print("\nGradient Norms:")
    
    for param, norm in grad_norms.items():
        print(f"  {param}: {norm:.6f}")
    
    # Print ratios if applicable
    if 'w1' in grad_norms and 'w2' in grad_norms and grad_norms['w1'] > 0:
        ratio = grad_norms['w2'] / grad_norms['w1']
        print(f"  w2/w1 ratio: {ratio:.3f}")


def extract_grad_norms(grad_info: dict) -> dict:
    """Extract gradient norms from grad_info, removing 'grad_' prefix."""
    return {k.replace('grad_', ''): v 
            for k, v in grad_info.items() 
            if k.startswith('grad_')}


def compute_activations(policy, obs):
    """Compute activations at each layer."""
    hidden_pre = obs @ policy.w1.value + policy.b1.value
    hidden_relu = jax.nn.relu(hidden_pre)
    
    if policy.use_layernorm and hasattr(policy, 'layer_norm'):
        hidden_post_ln = policy.layer_norm(hidden_relu)
        output = hidden_post_ln @ policy.w2.value + policy.b2.value
        return hidden_pre, hidden_relu, hidden_post_ln, output
    else:
        output = hidden_relu @ policy.w2.value + policy.b2.value
        return hidden_pre, hidden_relu, None, output


def compute_activation_stats(activations):
    """Compute statistics for each activation tensor."""
    hidden_pre, hidden_relu, hidden_ln, output = activations
    
    stats = {
        'hidden_pre_mean': float(jnp.mean(hidden_pre)),
        'hidden_pre_std': float(jnp.std(hidden_pre)),
        'hidden_relu_mean': float(jnp.mean(hidden_relu)),
        'hidden_relu_std': float(jnp.std(hidden_relu)),
        'output_mean': float(jnp.mean(output)),
        'output_std': float(jnp.std(output)),
    }
    
    if hidden_ln is not None:
        stats.update({
            'hidden_ln_mean': float(jnp.mean(hidden_ln)),
            'hidden_ln_std': float(jnp.std(hidden_ln)),
        })
    
    return stats