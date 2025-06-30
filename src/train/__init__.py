import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import NamedTuple
from flax import nnx
import optax
from ..baseline import BaselineState, update_baseline, compute_advantages
from ..advantage_normalizer import normalize_advantages
from ..critic import ValueFunction, update_critic, compute_critic_advantages
from .episodes import collect_episodes
from .metrics import MetricsTracker
from .lr_schedule import create_lr_schedule
from ..pendulum.features import compute_features


class TrainState(NamedTuple):
    step: int
    key: Array
    baseline: BaselineState


@nnx.jit
def train_step(
    policy,  # nnx.Module
    optimizer,  # nnx.Optimizer
    batch_states: Float[Array, "batch obs_dim"],
    batch_actions: Float[Array, "batch act_dim"],
    batch_advantages: Float[Array, "batch"],
    entropy_coef: float = 0.01,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    def loss_fn(policy):
        # Use policy's log_prob method
        log_probs = policy.log_prob(batch_states, batch_actions)
        
        # Get policy distribution for entropy
        mean, std = policy(batch_states)
        # Entropy of Gaussian: 0.5 * log(2 * pi * e * sigma^2)
        entropy = jnp.mean(0.5 * jnp.log(2 * jnp.pi * jnp.e * std**2))

        # REINFORCE loss with entropy bonus
        pg_loss = -jnp.mean(log_probs * batch_advantages)
        return pg_loss - entropy_coef * entropy

    loss, grads = nnx.value_and_grad(loss_fn)(policy)

    # Compute gradient statistics before update
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_norms = [
        jnp.linalg.norm(g.value if hasattr(g, "value") else g) for g in grad_leaves
    ]
    total_grad_norm = jnp.sqrt(sum(n**2 for n in grad_norms))
    grad_variance = jnp.var(
        jnp.concatenate(
            [
                g.value.flatten() if hasattr(g, "value") else g.flatten()
                for g in grad_leaves
            ]
        )
    )

    optimizer.update(grads)

    return loss, total_grad_norm, grad_variance


def train(
    policy,  # nnx.Module
    env_step,  # Environment step function
    env_reset,  # Environment reset function
    n_iterations: int = 100,
    episodes_per_iter: int = 10,
    learning_rate: float = 1e-3,
    warmup_steps: int = 100,
    use_baseline: bool = True,
    use_critic: bool = False,
    seed: int = 0,
    verbose: bool = True,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
) -> dict:
    # Initialize
    key = jax.random.PRNGKey(seed)
    
    # Create learning rate schedule
    lr_schedule = create_lr_schedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=n_iterations,
        end_lr_factor=0.1,
    )
    
    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosion
            optax.adam(lr_schedule, b1=adam_b1, b2=adam_b2),
        ),
    )

    # Initialize critic if requested
    if use_critic:
        critic = ValueFunction(obs_dim=2, hidden_dim=64)  # Uses raw 2D states
        critic_optimizer = nnx.Optimizer(
            critic, optax.adam(learning_rate * 2.0)
        )  # Faster critic learning
    else:
        critic = None
        critic_optimizer = None
        
    train_state = TrainState(
        step=0,
        key=key,
        baseline=BaselineState(mean=jnp.array(0.0), n_samples=0),
    )

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()

    for i in range(n_iterations):
        # Collect episodes in parallel
        key, collect_key = jax.random.split(train_state.key)
        train_state = train_state._replace(key=key)

        episode_batch = collect_episodes(
            policy, env_step, env_reset, collect_key, episodes_per_iter
        )

        # Data is already aggregated
        all_states = episode_batch.states  # Raw 2D states
        all_actions = episode_batch.actions
        all_returns = episode_batch.returns
        
        # Convert raw states to features for policy
        all_features = jax.vmap(compute_features)(all_states)

        # Compute advantages
        if use_critic:
            # Train critic to predict returns
            assert critic is not None and critic_optimizer is not None
            update_critic(critic, critic_optimizer, all_states, all_returns)
            # Use critic as baseline
            advantages = compute_critic_advantages(critic, all_states, all_returns)
        elif use_baseline:
            advantages = compute_advantages(all_returns, train_state.baseline.mean)
            new_baseline = update_baseline(train_state.baseline, all_returns)
            train_state = train_state._replace(baseline=new_baseline)
        else:
            advantages = all_returns

        # Store raw advantage std before normalization
        raw_adv_std = jnp.std(advantages)

        # Normalize advantages
        normalized_advantages = normalize_advantages(advantages)

        # Update policy
        loss, grad_norm, grad_var = train_step(
            policy, optimizer, all_features, all_actions, normalized_advantages
        )

        # Test policy at origin for tracking
        test_raw_state = jnp.zeros((1, 2))  # Test at origin (raw state)
        test_features = compute_features(test_raw_state[0])[None, :]
        test_mean, test_std = policy(test_features)
        
        # Compute policy entropy (average over batch)
        _, policy_stds = policy(all_features)
        # Entropy of Gaussian: 0.5 * log(2 * pi * e * sigma^2)
        entropy = jnp.mean(0.5 * jnp.log(2 * jnp.pi * jnp.e * policy_stds**2))
        
        # Get current learning rate
        current_lr = float(lr_schedule(i))
        
        # Update metrics
        metrics_tracker.update(
            iteration=i,
            episode_batch=episode_batch,
            loss=loss,
            grad_norm=grad_norm,
            grad_variance=grad_var,
            normalized_advantages=normalized_advantages,
            raw_advantage_std=raw_adv_std,
            baseline_mean=train_state.baseline.mean,
            policy_test_mean=test_mean,
            policy_test_std=test_std,
            episodes_per_iter=episodes_per_iter,
            learning_rate=current_lr,
            entropy=entropy,
        )
        
        # Log progress
        metrics_tracker.log_iteration(i, verbose=verbose)

    return metrics_tracker.to_dict()