import jax
from flax import nnx
from typing import Dict, List, Tuple
from tests.constants import DEFAULT_N_STEPS, LOG_INTERVAL
from tests.common_patterns import (
    create_loss_fn, extract_gradient_norms
)


@nnx.jit
def standard_train_step(
    policy: nnx.Module,
    optimizer: nnx.Optimizer,
    obs: jax.Array,
    targets: jax.Array
) -> Tuple[float, Dict[str, jax.Array]]:
    loss_fn = create_loss_fn(obs, targets)
    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    grad_norms = extract_gradient_norms(grads)
    optimizer.update(grads)
    return loss, grad_norms


def train_for_steps(
    policy: nnx.Module,
    optimizer: nnx.Optimizer,
    obs: jax.Array,
    targets: jax.Array,
    n_steps: int = DEFAULT_N_STEPS,
    log_interval: int = LOG_INTERVAL,
    track_gradients: bool = False,
    verbose: bool = True
) -> Dict[str, List[float]]:
    metrics = {'losses': []}
    if track_gradients:
        metrics['grad_norms'] = {
            'w1': [], 'b1': [], 'w2': [], 'b2': [], 'log_std': []
        }
    
    for step in range(n_steps):
        loss, grad_norms = standard_train_step(policy, optimizer, obs, targets)
        
        metrics['losses'].append(float(loss))
        
        if track_gradients:
            for name, norm in grad_norms.items():
                metrics['grad_norms'][name].append(float(norm))
        
        if verbose and step % log_interval == 0:
            print(f"Step {step}: loss = {loss:.6f}")
    
    return metrics


