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
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    def loss_fn(policy):
        # Use policy's log_prob method
        log_probs = policy.log_prob(batch_states, batch_actions)

        # REINFORCE loss
        return -jnp.mean(log_probs * batch_advantages)

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
    use_baseline: bool = True,
    use_critic: bool = False,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    # Initialize
    key = jax.random.PRNGKey(seed)
    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosion
            optax.adam(learning_rate),
        ),
    )

    # Initialize critic if requested
    if use_critic:
        critic = ValueFunction(obs_dim=2, hidden_dim=64)  # Pendulum-specific
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

    # Tracking
    metrics = {
        "iteration": [],
        "mean_return": [],
        "std_return": [],
        "loss": [],
        "baseline_value": [],
        "mean_advantage": [],
        "std_advantage": [],
        "mean_log_prob": [],
        "mean_action": [],
        "std_action": [],
        "grad_norm": [],
        "grad_variance": [],
        "raw_advantage_std": [],
        "episode_length": [],
        "grad_norm_clipped": [],
        "max_action": [],
        "min_action": [],
        "policy_mean_norm": [],
        "policy_std_mean": [],
        "returns_mean": [],
        "returns_std": [],
    }

    for i in range(n_iterations):
        # Collect episodes in parallel
        key, collect_key = jax.random.split(train_state.key)
        train_state = train_state._replace(key=key)

        episode_batch = collect_episodes(
            policy, env_step, env_reset, collect_key, episodes_per_iter
        )

        # Data is already aggregated
        all_states = episode_batch.states
        all_actions = episode_batch.actions
        all_returns = episode_batch.returns

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
        advantages = normalize_advantages(advantages)

        # Update policy
        loss, grad_norm, grad_var = train_step(
            policy, optimizer, all_states, all_actions, advantages
        )

        # Get log probs from the episode data
        log_probs = episode_batch.log_probs

        # Compute episode length (for debugging)
        episode_length = jnp.sum(episode_batch.rewards != 0.0) / episodes_per_iter

        # Track metrics (batched device transfer for efficiency)
        metric_array = jnp.array([
            episode_batch.total_reward,
            loss,
            train_state.baseline.mean / 400.0,  # Divide by episode length
            jnp.mean(advantages),
            jnp.std(advantages),
            jnp.mean(log_probs),
            jnp.mean(episode_batch.actions),
            jnp.std(episode_batch.actions),
            grad_norm,
            grad_var,
            raw_adv_std,
        ])
        metric_values = [float(x) for x in metric_array]
        
        metrics["iteration"].append(i)
        metrics["mean_return"].append(metric_values[0])  # This is average per step
        metrics["std_return"].append(0.0)  # TODO: track individual episode returns
        metrics["loss"].append(metric_values[1])
        metrics["baseline_value"].append(metric_values[2])
        metrics["mean_advantage"].append(metric_values[3])
        metrics["std_advantage"].append(metric_values[4])
        metrics["mean_log_prob"].append(metric_values[5])
        metrics["mean_action"].append(metric_values[6])
        metrics["std_action"].append(metric_values[7])
        metrics["grad_norm"].append(metric_values[8])
        metrics["grad_variance"].append(metric_values[9])
        metrics["raw_advantage_std"].append(metric_values[10])
        # Additional metrics (batched)
        test_states = jnp.zeros((1, 2))  # Test at origin
        test_mean, test_std = policy(test_states)
        
        additional_metrics = jnp.array([
            episode_length,
            jnp.minimum(grad_norm, 1.0),  # After clipping
            jnp.max(episode_batch.actions),
            jnp.min(episode_batch.actions),
            jnp.linalg.norm(test_mean),
            jnp.mean(test_std),
            jnp.mean(all_returns),
            jnp.std(all_returns),
        ])
        additional_values = [float(x) for x in additional_metrics]
        
        metrics["episode_length"].append(additional_values[0])
        metrics["grad_norm_clipped"].append(additional_values[1])
        metrics["max_action"].append(additional_values[2])
        metrics["min_action"].append(additional_values[3])
        metrics["policy_mean_norm"].append(additional_values[4])
        metrics["policy_std_mean"].append(additional_values[5])
        metrics["returns_mean"].append(additional_values[6])
        metrics["returns_std"].append(additional_values[7])

        if verbose and i % 10 == 0:
            print(
                f"Iter {i:3d} | Return: {metrics['mean_return'][-1]:6.3f} | "
                f"Loss: {metrics['loss'][-1]:10.2f} | "
                f"GradNorm: {metrics['grad_norm'][-1]:8.2f} | "
                f"RawAdvStd: {metrics['raw_advantage_std'][-1]:6.2f}"
            )

    return metrics